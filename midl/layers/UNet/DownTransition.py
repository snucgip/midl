"""
DownTransition layer proposed in UNet.
https://arxiv.org/abs/1505.04597

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn
from ..ConvBlock import ConvBlock
from ..Pool import Pool


class DownTransition(nn.Module):
    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None
    ):
        super(DownTransition, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.maxpool = Pool(dim=dim, kernel_size=2)
        self.conv1 = ConvBlock(
            dim=dim,
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = ConvBlock(
            dim=dim,
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

