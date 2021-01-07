"""
A wrapper for convolutional block layer

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn
from .Conv import Conv
from .BatchNorm import BN


class ConvBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'
    ):
        super(ConvBlock, self).__init__()

        self.conv = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        self.bn = BN(dim=dim, channels=out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
