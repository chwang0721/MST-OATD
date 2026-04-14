import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from logging_set import get_logger
from mst_oatd import MST_OATD
from utils import auc_score, make_mask, make_len_mask


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    seq_lengths = list(map(len, batch))
    batch_trajs = [x + [[0, [0] * 6]] * (max_len - len(x)) for x in batch]
    return torch.LongTensor(np.array(batch_trajs, dtype=object)[:, :, 0].tolist()), \
        torch.Tensor(np.array(batch_trajs, dtype=object)[:, :, 1].tolist()), np.array(seq_lengths)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        data_seqs = self.seqs[index]
        return data_seqs


def time_convert(times, time_interval):
    return torch.Tensor((times[:, :, 2] + times[:, :, 1] * 60 + times[:, :, 0] * 3600) // time_interval).long()


def savecheckpoint(state, file_name):
    torch.save(state, file_name)


class train_mst_oatd:
    def __init__(self, s_token_size, t_token_size, labels, train_loader, outliers_loader, args):

        self.MST_OATD_S = MST_OATD(s_token_size, s_token_size, args).to(args.device)
        self.MST_OATD_T = MST_OATD(s_token_size, t_token_size, args).to(args.device)

        self.device = args.device
        self.dataset = args.dataset
        self.n_cluster = args.n_cluster
        self.hidden_size = args.hidden_size

        self.crit = nn.CrossEntropyLoss()
        self.detec = nn.CrossEntropyLoss(reduction='none')

        self.gamma_params_s = [self.MST_OATD_S.pi_prior, self.MST_OATD_S.mu_prior, self.MST_OATD_S.log_var_prior]
        self.gamma_params_t = [self.MST_OATD_T.pi_prior, self.MST_OATD_T.mu_prior, self.MST_OATD_T.log_var_prior]

        gamma_ids_s = {id(param) for param in self.gamma_params_s}
        gamma_ids_t = {id(param) for param in self.gamma_params_t}
        self.phi_theta_params_s = [param for param in self.MST_OATD_S.parameters() if id(param) not in gamma_ids_s]
        self.phi_theta_params_t = [param for param in self.MST_OATD_T.parameters() if id(param) not in gamma_ids_t]

        self.pretrain_optimizer_s = optim.AdamW(self.phi_theta_params_s, lr=args.pretrain_lr_s)
        self.pretrain_optimizer_t = optim.AdamW(self.phi_theta_params_t, lr=args.pretrain_lr_t)

        self.phi_theta_optimizer_s = optim.AdamW(self.phi_theta_params_s, lr=args.lr_s)
        self.gamma_optimizer_s = optim.Adam(self.gamma_params_s, lr=args.lr_s)
        self.phi_theta_optimizer_t = optim.Adam(self.phi_theta_params_t, lr=args.lr_t)
        self.gamma_optimizer_t = optim.Adam(self.gamma_params_t, lr=args.lr_t)

        self.lr_pretrain_s = StepLR(self.pretrain_optimizer_s, step_size=2, gamma=0.9)
        self.lr_pretrain_t = StepLR(self.pretrain_optimizer_t, step_size=2, gamma=0.9)
        self.lr_phi_theta_s = StepLR(self.phi_theta_optimizer_s, step_size=2, gamma=0.9)
        self.lr_gamma_s = StepLR(self.gamma_optimizer_s, step_size=2, gamma=0.9)
        self.lr_phi_theta_t = StepLR(self.phi_theta_optimizer_t, step_size=2, gamma=0.9)
        self.lr_gamma_t = StepLR(self.gamma_optimizer_t, step_size=2, gamma=0.9)

        self.train_loader = train_loader
        self.outliers_loader = outliers_loader

        self.pretrained_path = 'models/pretrain_mstoatd_{}.pth'.format(args.dataset)
        self.path_checkpoint = 'models/mstoatd_{}.pth'.format(args.dataset)
        self.logger = get_logger("./logs/{}.log".format(args.dataset))

        self.labels = labels
        if args.dataset == 'cd':
            self.time_interval = 10
        else:
            self.time_interval = 15
        self.mode = 'train'

        self.s1_size = args.s1_size
        self.s2_size = args.s2_size

    def set_parameter_state(self, phi_theta_grad, gamma_grad):
        for param in self.phi_theta_params_s + self.phi_theta_params_t:
            param.requires_grad_(phi_theta_grad)
        for param in self.gamma_params_s + self.gamma_params_t:
            param.requires_grad_(gamma_grad)

    def pretrain(self, epoch):
        self.MST_OATD_S.train()
        self.MST_OATD_T.train()
        self.set_parameter_state(True, False)
        epo_loss = 0

        for batch in self.train_loader:
            trajs, times, seq_lengths = batch
            batch_size = len(trajs)

            mask = make_mask(make_len_mask(trajs)).to(self.device)

            self.pretrain_optimizer_s.zero_grad()
            self.pretrain_optimizer_t.zero_grad()
            output_s, _, _, _ = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
            output_t, _, _, _ = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "pretrain", -1)

            times = time_convert(times, self.time_interval)

            loss = self.crit(output_s[mask == 1], trajs.to(self.device)[mask == 1])
            loss += self.crit(output_t[mask == 1], times.to(self.device)[mask == 1])

            loss.backward()

            self.pretrain_optimizer_s.step()
            self.pretrain_optimizer_t.step()
            epo_loss += loss.item()

        self.lr_pretrain_s.step()
        self.lr_pretrain_t.step()
        self.set_parameter_state(True, True)
        epo_loss = "%.4f" % (epo_loss / len(self.train_loader))
        self.logger.info("Epoch {} pretrain loss: {}".format(epoch + 1, epo_loss))
        checkpoint = {"model_state_dict_s": self.MST_OATD_S.state_dict(),
                      "model_state_dict_t": self.MST_OATD_T.state_dict()}
        torch.save(checkpoint, self.pretrained_path)

    def train(self, epoch):
        self.MST_OATD_S.train()
        self.MST_OATD_T.train()
        total_loss = 0
        total_recon = 0
        total_kl_c = 0
        total_kl_r = 0
        for batch in self.train_loader:
            trajs, times, seq_lengths = batch
            batch_size = len(trajs)
            mask = make_mask(make_len_mask(trajs)).to(self.device)
            target_times = time_convert(times, self.time_interval)

            self.set_parameter_state(True, False)
            self.phi_theta_optimizer_s.zero_grad()
            self.phi_theta_optimizer_t.zero_grad()

            x_hat_s, mu_s, log_var_s, z_s = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "train", -1)
            loss_s_phi, stats_s_phi = self.Loss(
                x_hat_s, trajs.to(self.device), mu_s.squeeze(0), log_var_s.squeeze(0), z_s.squeeze(0),
                self.MST_OATD_S, mask
            )
            x_hat_t, mu_t, log_var_t, z_t = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "train", -1)
            loss_t_phi, stats_t_phi = self.Loss(
                x_hat_t, target_times.to(self.device), mu_t.squeeze(0), log_var_t.squeeze(0), z_t.squeeze(0),
                self.MST_OATD_T, mask
            )
            loss_phi_theta = loss_s_phi + loss_t_phi
            loss_phi_theta.backward()
            self.phi_theta_optimizer_s.step()
            self.phi_theta_optimizer_t.step()

            self.set_parameter_state(False, True)
            self.gamma_optimizer_s.zero_grad()
            self.gamma_optimizer_t.zero_grad()

            x_hat_s, mu_s, log_var_s, z_s = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "train", -1)
            loss_s_gamma, stats_s_gamma = self.Loss(
                x_hat_s, trajs.to(self.device), mu_s.squeeze(0), log_var_s.squeeze(0), z_s.squeeze(0),
                self.MST_OATD_S, mask
            )
            x_hat_t, mu_t, log_var_t, z_t = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "train", -1)
            loss_t_gamma, stats_t_gamma = self.Loss(
                x_hat_t, target_times.to(self.device), mu_t.squeeze(0), log_var_t.squeeze(0), z_t.squeeze(0),
                self.MST_OATD_T, mask
            )
            loss_gamma = loss_s_gamma + loss_t_gamma
            loss_gamma.backward()
            self.gamma_optimizer_s.step()
            self.gamma_optimizer_t.step()

            batch_loss = 0.5 * (loss_phi_theta.item() + loss_gamma.item())
            total_loss += batch_loss
            total_recon += 0.5 * (stats_s_phi['reconstruction'] + stats_t_phi['reconstruction'] +
                                  stats_s_gamma['reconstruction'] + stats_t_gamma['reconstruction'])
            total_kl_c += 0.5 * (stats_s_phi['category_kl'] + stats_t_phi['category_kl']
                                 + stats_s_gamma['category_kl'] + stats_t_gamma['category_kl'])
            total_kl_r += 0.5 * (stats_s_phi['gaussian_kl'] + stats_t_phi['gaussian_kl']
                                 + stats_s_gamma['gaussian_kl'] + stats_t_gamma['gaussian_kl'])

        self.set_parameter_state(True, True)
        self.lr_phi_theta_s.step()
        self.lr_gamma_s.step()
        self.lr_phi_theta_t.step()
        self.lr_gamma_t.step()

        if self.mode == "train":
            num_batches = len(self.train_loader)
            self.logger.info(
                'Epoch {} loss: {:.4f}, recon: {:.4f}, kl(c): {:.4f}, kl(r): {:.4f}'.format(
                    epoch + 1,
                    total_loss / num_batches,
                    total_recon / num_batches,
                    total_kl_c / num_batches,
                    total_kl_r / num_batches,
                )
            )
            checkpoint = {"model_state_dict_s": self.MST_OATD_S.state_dict(),
                          "model_state_dict_t": self.MST_OATD_T.state_dict()}
            torch.save(checkpoint, self.path_checkpoint)

    def pretrain_detection(self):
        self.MST_OATD_S.eval()
        self.MST_OATD_T.eval()

        all_likelihood_s = []
        all_likelihood_t = []

        with torch.no_grad():
            for batch in self.outliers_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                mask = make_mask(make_len_mask(trajs)).to(self.device)
                times_token = time_convert(times, self.time_interval)

                output_s, _, _, _ = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                likelihood_s = -self.detec(output_s.reshape(-1, output_s.shape[-1]),
                                           trajs.to(self.device).reshape(-1))
                likelihood_s = torch.exp(
                    torch.sum(mask * likelihood_s.reshape(batch_size, -1), dim=-1) / torch.sum(mask, 1))
                all_likelihood_s.append(likelihood_s)

                output_t, _, _, _ = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                likelihood_t = -self.detec(output_t.reshape(-1, output_t.shape[-1]),
                                           times_token.to(self.device).reshape(-1))
                likelihood_t = torch.exp(
                    torch.sum(mask * likelihood_t.reshape(batch_size, -1), dim=-1) / torch.sum(mask, 1))
                all_likelihood_t.append(likelihood_t)

        likelihood_s = torch.cat(all_likelihood_s, dim=0)
        likelihood_t = torch.cat(all_likelihood_t, dim=0)

        pr_auc = auc_score(self.labels, (1 - likelihood_s * likelihood_t).cpu().detach().numpy())
        self.logger.info(f'Pretrain PR_AUC: {pr_auc:.6f}')
        return pr_auc

    def detection(self):

        self.MST_OATD_S.eval()
        all_likelihood_s = []
        self.MST_OATD_T.eval()
        all_likelihood_t = []

        with torch.no_grad():

            for batch in self.outliers_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                mask = make_mask(make_len_mask(trajs)).to(self.device)
                times_token = time_convert(times, self.time_interval)

                c_likelihood_s = []
                c_likelihood_t = []

                for c in range(self.n_cluster):
                    output_s, _, _, _ = self.MST_OATD_S(trajs, times, seq_lengths, batch_size, "test", c)
                    likelihood_s = - self.detec(output_s.reshape(-1, output_s.shape[-1]),
                                                trajs.to(self.device).reshape(-1))
                    likelihood_s = torch.exp(
                        torch.sum(mask * (likelihood_s.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    output_t, _, _, _ = self.MST_OATD_T(trajs, times, seq_lengths, batch_size, "test", c)
                    likelihood_t = - self.detec(output_t.reshape(-1, output_t.shape[-1]),
                                                times_token.to(self.device).reshape(-1))
                    likelihood_t = torch.exp(
                        torch.sum(mask * (likelihood_t.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    c_likelihood_s.append(likelihood_s.unsqueeze(0))
                    c_likelihood_t.append(likelihood_t.unsqueeze(0))

                all_likelihood_s.append(torch.cat(c_likelihood_s).max(0)[0])
                all_likelihood_t.append(torch.cat(c_likelihood_t).max(0)[0])

        likelihood_s = torch.cat(all_likelihood_s, dim=0)
        likelihood_t = torch.cat(all_likelihood_t, dim=0)

        pr_auc = auc_score(self.labels, (1 - likelihood_s * likelihood_t).cpu().detach().numpy())
        return pr_auc

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_vars[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def route_type_posterior(self, model, z):
        log_pi = F.log_softmax(model.pi_prior, dim=-1)
        log_q_c = log_pi.unsqueeze(0) + self.gaussian_pdfs_log(z, model.mu_prior, model.log_var_prior)
        q_c = F.softmax(log_q_c, dim=-1).clamp_min(1e-10)
        q_c = q_c / q_c.sum(dim=-1, keepdim=True)
        return q_c, log_pi

    def Loss(self, x_hat, targets, z_mu, z_log_var, z, model, mask):
        reconstruction_loss = self.crit(x_hat[mask == 1], targets[mask == 1])

        q_c, log_pi = self.route_type_posterior(model, z)
        category_loss = torch.sum(q_c * (torch.log(q_c) - log_pi.unsqueeze(0)), dim=-1).mean()

        mu_c = model.mu_prior.unsqueeze(0)
        log_var_c = model.log_var_prior.unsqueeze(0)
        posterior_mu = z_mu.unsqueeze(1)
        posterior_log_var = z_log_var.unsqueeze(1)
        posterior_var = torch.exp(posterior_log_var)
        prior_var = torch.exp(log_var_c)

        kl_r_c = 0.5 * torch.sum(log_var_c - posterior_log_var
                                 + (posterior_var + (posterior_mu - mu_c).pow(2)) / (prior_var + 1e-10) - 1, dim=-1)
        gaussian_loss = torch.sum(q_c * kl_r_c, dim=-1).mean()

        loss = reconstruction_loss + category_loss + gaussian_loss
        stats = {
            'reconstruction': reconstruction_loss.item(),
            'category_kl': category_loss.item(),
            'gaussian_kl': gaussian_loss.item(),
        }
        return loss, stats

    def load_pretrained(self):
        checkpoint = torch.load(self.pretrained_path)
        self.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])
