"""
A Dice Loss implementation

Author      : Sanguk Park
Version     : 0.1
Source      : https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
"""

import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        numerator = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        denominator = torch.sum(torch.add(predict, target), dim=1) + self.smooth

        loss = 1 - numerator / denominator

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target, weight):
        assert(predict.shape == target.shape)
        dice = BinaryDiceLoss()

        total_loss = 0.0
        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            dice_loss *= weight[i]

            total_loss += dice_loss

        return total_loss / target.shape[1]

