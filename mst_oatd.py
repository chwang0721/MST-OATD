import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from temporal import TemporalEmbedding


class co_attention(nn.Module):
    def __init__(self, dim):
        super(co_attention, self).__init__()

        self.Wq_s = nn.Linear(dim, dim, bias=False)
        self.Wk_s = nn.Linear(dim, dim, bias=False)
        self.Wv_s = nn.Linear(dim, dim, bias=False)

        self.Wq_t = nn.Linear(dim, dim, bias=False)
        self.Wk_t = nn.Linear(dim, dim, bias=False)
        self.Wv_t = nn.Linear(dim, dim, bias=False)

        self.dim_k = dim ** 0.5

        self.FFN_s = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.FFN_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t):
        seq_t = seq_t.unsqueeze(2)
        seq_s = seq_s.unsqueeze(2)

        q_s, k_s, v_s = self.Wq_s(seq_t), self.Wk_s(seq_s), self.Wv_s(seq_s)
        q_t, k_t, v_t = self.Wq_t(seq_s), self.Wk_t(seq_t), self.Wv_t(seq_t)

        coatt_s = F.softmax(torch.matmul(q_s / self.dim_k, k_s.transpose(2, 3)), dim=-1)
        coatt_t = F.softmax(torch.matmul(q_t / self.dim_k, k_t.transpose(2, 3)), dim=-1)

        att_s = self.layer_norm(self.FFN_s(torch.matmul(coatt_s, v_s)) + torch.matmul(coatt_s, v_s))
        att_t = self.layer_norm(self.FFN_t(torch.matmul(coatt_t, v_t)) + torch.matmul(coatt_t, v_t))

        return att_s.squeeze(2), att_t.squeeze(2)


class state_attention(nn.Module):
    def __init__(self, args):
        super(state_attention, self).__init__()

        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, seq):
        u = torch.tanh(torch.matmul(seq, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = seq * att_score
        return scored_outputs.sum(1)


class MST_OATD(nn.Module):
    def __init__(self, token_size, token_size_out, args):
        super(MST_OATD, self).__init__()

        self.emb_size = args.embedding_size
        self.device = args.device
        self.n_cluster = args.n_cluster
        self.dataset = args.dataset
        self.s1_size = args.s1_size
        self.s2_size = args.s2_size

        self.pi_prior = nn.Parameter(torch.ones(args.n_cluster) / args.n_cluster)
        self.mu_prior = nn.Parameter(torch.randn(args.n_cluster, args.hidden_size))
        self.log_var_prior = nn.Parameter(torch.zeros(args.n_cluster, args.hidden_size))

        self.embedding = nn.Embedding(token_size, args.embedding_size)

        self.encoder_s1 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.encoder_s2 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.encoder_s3 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)

        self.decoder = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)

        self.fc_mu = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_logvar = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)

        self.fc_out = nn.Linear(args.hidden_size, token_size_out)

        self.nodes = torch.arange(token_size, dtype=torch.long).to(args.device)
        self.adj = sparse.load_npz("data/{}/adj.npz".format(args.dataset))
        self.d_norm = sparse.load_npz("data/{}/d_norm.npz".format(args.dataset))

        if args.dataset == 'porto':
            self.V = nn.Parameter(torch.Tensor(token_size, token_size))
        else:
            self.V = nn.Parameter(torch.Tensor(args.embedding_size, args.embedding_size))

        self.W1 = nn.Parameter(torch.ones(1) / 3)
        self.W2 = nn.Parameter(torch.ones(1) / 3)
        self.W3 = nn.Parameter(torch.ones(1) / 3)

        self.co_attention = co_attention(args.embedding_size).to(args.device)
        self.d2v = TemporalEmbedding(args.device)

        self.w_omega = nn.Parameter(torch.Tensor(args.embedding_size * 2, args.embedding_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(args.embedding_size * 2, 1))

        nn.init.uniform_(self.V, -0.2, 0.2)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.state_att = state_attention(args)
        self.dataset = args.dataset

    def scale_process(self, e_inputs, scale_size, lengths):
        e_inputs_split = torch.mean(e_inputs.unfold(1, scale_size, scale_size), dim=3)
        e_inputs_split = self.attention_layer(e_inputs_split, lengths)
        e_inputs_split = pack_padded_sequence(e_inputs_split, lengths, batch_first=True, enforce_sorted=False)
        return e_inputs_split

    def Norm_A(self, A, D):
        return D.mm(A).mm(self.V).mm(D)

    def Norm_A_N(self, A, D):
        return D.mm(A).mm(D)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def padding_mask(self, inp):
        return inp == 0

    def attention_layer(self, e_input, lengths):
        mask = self.getMask(lengths)
        u = torch.tanh(torch.matmul(e_input, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(mask == 0, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        att_e_input = e_input * att_score
        return att_e_input

    def array2sparse(self, A):
        A = A.tocoo()
        values = A.data
        indices = np.vstack((A.row, A.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(values).to(self.device)
        A = torch.sparse_coo_tensor(i, v, torch.Size(A.shape), dtype=torch.float32)
        return A

    def getMask(self, seq_lengths):
        max_len = max(seq_lengths)
        mask = torch.ones((len(seq_lengths), max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, trajs, times, lengths, batch_size, mode, c):

        # spatial embedding
        adj = self.array2sparse(self.adj)
        d_norm = self.array2sparse(self.d_norm)
        if self.dataset == 'porto':
            H = self.Norm_A(adj, d_norm)
            nodes = H.mm(self.embedding(self.nodes))
        else:
            H = self.Norm_A_N(adj, d_norm)
            nodes = H.mm(self.embedding(self.nodes)).mm(self.V)

        s_inputs = torch.index_select(nodes, 0, trajs.flatten().to(self.device)). \
            reshape(batch_size, -1, self.emb_size)

        # temporal embedding
        t_inputs = self.d2v(times.to(self.device)).to(self.device)

        att_s, att_t = self.co_attention(s_inputs, t_inputs)
        st_inputs = torch.concat((att_s, att_t), dim=2)
        d_inputs = torch.cat((torch.zeros(batch_size, 1, self.emb_size * 2, dtype=torch.long).to(self.device),
                              st_inputs[:, :-1, :]), dim=1)  # [bs, seq_len, emb_size * 2]

        decoder_inputs = pack_padded_sequence(d_inputs, lengths, batch_first=True, enforce_sorted=False)

        if mode == "pretrain" or "train":
            encoder_inputs_s1 = pack_padded_sequence(self.attention_layer(st_inputs, lengths), lengths,
                                                     batch_first=True, enforce_sorted=False)
            encoder_inputs_s2 = self.scale_process(st_inputs, self.s1_size, [int(i // self.s1_size) for i in lengths])
            encoder_inputs_s3 = self.scale_process(st_inputs, self.s2_size, [int(i // self.s2_size) for i in lengths])

            _, encoder_final_state_s1 = self.encoder_s1(encoder_inputs_s1)
            _, encoder_final_state_s2 = self.encoder_s2(encoder_inputs_s2)
            _, encoder_final_state_s3 = self.encoder_s3(encoder_inputs_s3)

            encoder_final_state = (self.W1 * encoder_final_state_s1 + self.W2 * encoder_final_state_s2
                                   + self.W3 * encoder_final_state_s3)
            sum_W = self.W1.data + self.W2.data + self.W3.data
            self.W1.data /= sum_W
            self.W2.data /= sum_W
            self.W3.data /= sum_W

            mu = self.fc_mu(encoder_final_state)
            logvar = self.fc_logvar(encoder_final_state)
            z = self.reparameterize(mu, logvar)

            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)

        elif mode == "test":
            mu = torch.stack([self.mu_prior] * batch_size, dim=1)[c: c + 1]
            decoder_outputs, _ = self.decoder(decoder_inputs, mu)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            logvar, z = None, None

        output = self.fc_out(self.layer_norm(decoder_outputs))

        return output, mu, logvar, z
