import torch
import torch.nn as nn
import torch.nn.functional as F


class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
        
    def forward(self, pred_y, y):
        mse = F.mse_loss(pred_y, y, reduction='mean')
        
        return mse


class huber_loss(nn.Module):
    def __init__(self, delta=1.0):
        super(huber_loss, self).__init__()
        self.delta = delta
        
    def forward(self, pred_y, y):
        h = F.huber_loss(pred_y, y, reduction='mean', delta=self.delta)
        
        return h


def ncc_loss(pred_y, y, eps=1e-8):
    N = y.shape[0]

    n_y = (y - torch.mean(y, 0, True)) / (torch.std(y, 0, True) + eps)
    n_pred_y = (pred_y - torch.mean(pred_y, 0, True)) / (torch.std(pred_y, 0, True) + eps)

    n_loss = - torch.sum(n_y * n_pred_y) / (N-1)

    return n_loss

