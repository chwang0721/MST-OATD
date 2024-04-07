import torch
from sklearn.metrics import average_precision_score


def make_len_mask(inp):
    return inp == 0


def make_mask(mask):
    return (~mask).detach().type(torch.uint8)


def auc_score(y_true, y_score):
    # precision, recall, _ = precision_recall_curve(y_true, y_score)
    # return auc(recall, precision)
    return average_precision_score(y_true, y_score)

