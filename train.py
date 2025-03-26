import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from mst_oatd_trainer import train_mst_oatd, MyDataset, seed_torch, collate_fn


def main():
    train_trajs = np.load('./data/{}/train_data_init.npy'.format(args.dataset), allow_pickle=True)
    test_trajs = np.load('./data/{}/outliers_data_init_{}_{}_{}.npy'.format(args.dataset, args.distance, args.fraction,
                                                                            args.obeserved_ratio), allow_pickle=True)
    outliers_idx = np.load("./data/{}/outliers_idx_init_{}_{}_{}.npy".format(args.dataset, args.distance, args.fraction,
                                                                             args.obeserved_ratio), allow_pickle=True)

    train_data = MyDataset(train_trajs)
    test_data = MyDataset(test_trajs)

    labels = np.zeros(len(test_trajs))
    for i in outliers_idx:
        labels[i] = 1
    labels = labels

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=8, pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=8, pin_memory=True)

    MST_OATD = train_mst_oatd(s_token_size, t_token_size, labels, train_loader, outliers_loader, args)

    if args.task == 'train':

        MST_OATD.logger.info("Start pretraining!")

        for epoch in range(args.pretrain_epochs):
            MST_OATD.pretrain(epoch)

        MST_OATD.train_gmm()
        MST_OATD.save_weights_for_MSTOATD()

        MST_OATD.logger.info("Start training!")
        MST_OATD.load_mst_oatd()
        for epoch in range(args.epochs):
            MST_OATD.train(epoch)

    if args.task == 'test':

        MST_OATD.logger.info('Start testing!')
        MST_OATD.logger.info("d = {}".format(args.distance) + ", " + chr(945) + " = {}".format(args.fraction) + ", "
              + chr(961) + " = {}".format(args.obeserved_ratio))

        checkpoint = torch.load(MST_OATD.path_checkpoint, weights_only=False)
        MST_OATD.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        MST_OATD.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])
        pr_auc = MST_OATD.detection()
        pr_auc = "%.4f" % pr_auc
        MST_OATD.logger.info("PR_AUC: {}".format(pr_auc))

    if args.task == 'train':
        MST_OATD.train_gmm_update()
        z = MST_OATD.get_hidden()
        MST_OATD.get_prob(z.cpu())


if __name__ == "__main__":

    if args.dataset == 'porto':
        s_token_size = 51 * 119
        t_token_size = 5760

    elif args.dataset == 'cd':
        s_token_size = 167 * 154
        t_token_size = 8640

    main()
