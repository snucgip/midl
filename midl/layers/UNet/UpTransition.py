"""
UpTransition layer proposed in UNet.
https://arxiv.org/abs/1505.04597

Author      : Sanguk Park
Version     : 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Conv import ConvTranspose
from ..ConvBlock import ConvBlock

class UpTransition(nn.Module):
    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            bilinear=True
    ):

        super(UpTransition, self).__init__()
        if bilinear:
            if dim == 2:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            elif dim == 3:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv1 = ConvBlock(dim=dim,
                                   in_channels=in_channels,
                                   out_channels=in_channels//2,
                                   kernel_size=3,
                                   padding=1)
            self.conv2 = ConvBlock(dim=dim,
                                   in_channels=in_channels//2,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1)
        else:
            self.up = ConvTranspose(dim=dim,
                                    in_channels=in_channels,
                                    out_channels=in_channels//2,
                                    kernel_size=2,
                                    stride=2)
            self.conv1 = ConvBlock(dim=dim,
                                   in_channels=in_channels,
                                   out_channels=in_channels // 2,
                                   kernel_size=3,
                                   padding=1)
            self.conv2 = ConvBlock(dim=dim,
                                   in_channels=in_channels // 2,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1)

    def forward(self, x, skipx):
        x = self.up(x)

        diffY = skipx.size()[2] - x.size()[2]
        diffX = skipx.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat([skipx, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x
