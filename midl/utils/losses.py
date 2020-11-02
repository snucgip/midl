"""
File:		Loss Layer (custom)
Language:	Python 3.6.5
Library:	PyTorch 0.4.0
Author:		Minyoung Chung
Date:		2018-05-06
Version:	1.0
Mail:		chungmy.freddie@gmail.com

    " Dice loss reference: https://github.com/mattmacy/torchbiomed/blob/master/torchbiomed/loss.py "

Copyright (c) 2018 All rights reserved by Bokkmani Studio corp.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Function
from itertools import repeat
import numpy as np

def penalty_dice(x, label, penalty):
    # make channels the last axis
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    # flatten
    x = x.view(x.numel() // 2, 2)
    x = F.softmax(x)
    prob = x[:, 1]

    penalty = penalty.permute(0, 2, 3, 4, 1).contiguous()
    penalty = penalty.view(penalty.numel())

    label = label.permute(0, 2, 3, 4, 1).contiguous()
    label = label.view(label.numel())
    label = label.float()

    label = label * penalty

    return 1-dice(prob, label)

# inverse included from dice2.
def dice3(x, label):
    # make channels the last axis
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    # flatten
    x = x.view(x.numel() // 2, 2)
    x = F.softmax(x)
    prob_liver = x[:, 1]
    prob_background = x[:, 0]

    label = label.permute(0, 2, 3, 4, 1).contiguous()
    label = label.view(label.numel())

    label_inv = label.clone()
    label_inv[label==0] = 1
    label_inv[label!=0] = 0

    return 2 - dice(prob_liver, label.float()) - dice(prob_background, label_inv.float())

def dice_inv(x, label):
    # make channels the last axis
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    # flatten
    x = x.view(x.numel() // 2, 2)
    x = F.softmax(x)
    prob_background = x[:, 0]

    label = label.permute(0, 2, 3, 4, 1).contiguous()
    label = label.view(label.numel())

    label_inv = label.clone()
    label_inv[label==0] = 1
    label_inv[label!=0] = 0

    return 1 - dice(prob_background, label_inv.float())

# X-entropy foramatted input version.
def dice2(x, label, hard=False):
    # make channels the last axis
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    # flatten
    x = x.view(x.numel() // 2, 2)
    x = F.softmax(x)
    prob = x[:, 1]
    if hard:
        prob = (prob >= 0.5).float() * 1

    label = label.permute(0, 2, 3, 4, 1).contiguous()
    label = label.view(label.numel())

    return 1 - dice(prob, label.float())

def dice(input, target):
    intersect = torch.dot(input, target)
    union = torch.sum(input) + torch.sum(target) + (2 * 0.000001)
    IoU = intersect / union
    return 2 * IoU


# def dice2_soft(x, label):
#     # make channels the last axis
#     x = x.permute(0, 2, 3, 4, 1).contiguous()
#     # flatten
#     x = x.view(x.numel() // 2, 2)
#     x = F.softmax(x)
#     prob = x[:, 1]
#
#     label = label.permute(0, 2, 3, 4, 1).contiguous()
#     label = label.view(label.numel())
#
#     return 1 - dice_soft(prob, label.float())
#
# def dice_soft(input, target):
#     over = torch.dot(input, target)
#     denom = torch.sqrt(torch.sum(torch.pow(input, 2))) + torch.sqrt(torch.sum(torch.pow(target, 2)))
#     return over / denom


# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]
# class DiceLoss(Function):
#     def forward(ctx, input, target):
#         eps = 0.000001
#         _, result_ = input.max(1)
#         result_ = torch.squeeze(result_)
#         if input.is_cuda:
#             result = torch.cuda.FloatTensor(result_.size())
#             target_ = torch.cuda.FloatTensor(target.size())
#         else:
#             result = torch.FloatTensor(result_.size())
#             target_ = torch.FloatTensor(target.size())
#         result.copy_(result_)
#         target_.copy_(target)
#         target = target_
#         #       print(input)
#         intersect = torch.dot(result, target)
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2 * eps)
#
#         ctx.save_for_backward(input, result, target, intersect, union)
#
#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#         #     union, intersect, target_sum, result_sum, 2 * IoU))
#         out = torch.FloatTensor(1).fill_(2 * IoU)
#
#         return out
#
#     def backward(ctx, grad_output):
#         # pack = ctx.saved_tensors
#         input, result, target, intersect, union = ctx.saved_tensors
#
#         # strange error... move to gpu manually.
#         grad_output = grad_output.cuda()
#
#         gt = torch.div(target, union)
#         IoU2 = intersect / (union * union)
#         # pred = torch.mul(result, IoU2)
#         pred = torch.mul(input[:,1], IoU2)
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), 0)
#         return grad_input, None
#
#
# def dice_loss(input, target):
#     return DiceLoss()(input, target)
#
#
# """
#     https://github.com/mattmacy/torchbiomed/blob/master/torchbiomed/loss.py
# """
# # import torch
# # from torch.autograd import Function
# # from itertools import repeat
# # import numpy as np
# #
# # # Intersection = dot(A, B)
# # # Union = dot(A, A) + dot(B, B)
# # # The Dice loss function is defined as
# # # 1/2 * intersection / union
# # #
# # # The derivative is 2[(union * target - 2 * intersect * input) / union^2]
# #
# class DiceLossRef(Function):
#     def __init__(self, *args, **kwargs):
#         pass
#
#     def forward(self, input, target, save=True):
#         if save:
#             self.save_for_backward(input, target)
#         eps = 0.000001
#         _, result_ = input.max(1)
#         result_ = torch.squeeze(result_)
#         if input.is_cuda:
#             result = torch.cuda.FloatTensor(result_.size())
#             self.target_ = torch.cuda.FloatTensor(target.size())
#         else:
#             result = torch.FloatTensor(result_.size())
#             self.target_ = torch.FloatTensor(target.size())
#         result.copy_(result_)
#         self.target_.copy_(target)
#         target = self.target_
# #       print(input)
#         intersect = torch.dot(result, target)
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2*eps)
#
#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#         #     union, intersect, target_sum, result_sum, 2*IoU))
#         out = torch.FloatTensor(1).fill_(2*IoU)
#         self.intersect, self.union = intersect, union
#         return out
#
#     def backward(self, grad_output):
#         # strange error... move to gpu manually.
#         grad_output = grad_output.cuda()
#
#         input, _ = self.saved_tensors
#         intersect, union = self.intersect, self.union
#         target = self.target_
#         gt = torch.div(target, union)
#         IoU2 = intersect/(union*union)
#         pred = torch.mul(input[:, 1], IoU2)
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), 0)
#         return grad_input , None
# #
# # def dice_loss(input, target):
# #     return DiceLoss()(input, target)
# #
# # def dice_error(input, target):
# #     eps = 0.000001
# #     _, result_ = input.max(1)
# #     result_ = torch.squeeze(result_)
# #     if input.is_cuda:
# #         result = torch.cuda.FloatTensor(result_.size())
# #         target_ = torch.cuda.FloatTensor(target.size())
# #     else:
# #         result = torch.FloatTensor(result_.size())
# #         target_ = torch.FloatTensor(target.size())
# #     result.copy_(result_.data)
# #     target_.copy_(target.data)
# #     target = target_
# #     intersect = torch.dot(result, target)
# #
# #     result_sum = torch.sum(result)
# #     target_sum = torch.sum(target)
# #     union = result_sum + target_sum + 2*eps
# #     intersect = np.max([eps, intersect])
# #     # the target volume can be empty - so we still want to
# #     # end up with a score of 1 if the result is 0/0
# #     IoU = intersect / union
# # #    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
# # #        union, intersect, target_sum, result_sum, 2*IoU))
# #     return 2*IoU
