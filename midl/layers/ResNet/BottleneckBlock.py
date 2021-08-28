"""
A BottleneckBlock is a module from ResNet.

Author      : Sanguk Park
Version     : 0.1
"""

import torch
import torch.nn as nn

from ..Conv import Conv
from ..BatchNorm import BN


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, dim, in_channels, planes):
        super(BottleneckBlock, self).__init__()

        self.conv1 = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=planes,
            kernel_size=1,
            bias=False
        )
        self.bn1 = BN(dim=dim, channels=planes)
        self.act1 = nn.ReLU()
        self.conv2 = Conv(
            dim=dim,
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = BN(dim=dim, channels=planes)
        self.act2 = nn.ReLU()
        self.conv3 = Conv(
            dim=dim,
            in_channels=planes,
            out_channels=self.expansion * planes,
            kernel_size=1,
            bias=False
        )
        self.bn3 = BN(dim=dim, channels=self.expansion * planes)
        self.act3 = nn.ReLU()

    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += skip
        out = self.act3(out)
        return out
