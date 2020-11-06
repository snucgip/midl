"""
An implementation for U-Net
https://arxiv.org/abs/1505.04597

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn

from midl.layers.Conv import Conv
from midl.layers.ConvBlock import ConvBlock
from midl.layers.UNet.DownTransition import DownTransition
from midl.layers.UNet.UpTransition import UpTransition

class UNet(nn.Module):
    def __init__(
            self,
            dim,
            in_channels,
            n_classes,
            bilinear=True
    ):
        super(UNet, self).__init__()

        if dim == 2:
            out_shapes = [64, 128, 256, 512, 1024]
        elif dim == 3:
            out_shapes = [32, 64, 128, 256, 512]
        else:
            raise ValueError()

        self.in_channels = in_channels
        self.n_clasees = n_classes
        self.bilinear = bilinear

        self.in_tr = nn.Sequential(
            ConvBlock(dim=dim, in_channels=in_channels, out_channels=out_shapes[0], kernel_size=3, padding=1),
            ConvBlock(dim=dim, in_channels=out_shapes[0], out_channels=out_shapes[0], kernel_size=3, padding=1)
        )

        self.down1 = DownTransition(dim=dim, in_channels=out_shapes[0], out_channels=out_shapes[1])
        self.down2 = DownTransition(dim=dim, in_channels=out_shapes[1], out_channels=out_shapes[2])
        self.down3 = DownTransition(dim=dim, in_channels=out_shapes[2], out_channels=out_shapes[3])

        factor = 2 if bilinear else 1
        self.down4 = DownTransition(dim=dim, in_channels=out_shapes[3], out_channels=out_shapes[4]//factor)

        self.up1 = UpTransition(dim=dim, in_channels=out_shapes[4], out_channels=out_shapes[3]//factor, bilinear=bilinear)
        self.up2 = UpTransition(dim=dim, in_channels=out_shapes[3], out_channels=out_shapes[2]//factor, bilinear=bilinear)
        self.up3 = UpTransition(dim=dim, in_channels=out_shapes[2], out_channels=out_shapes[1]//factor, bilinear=bilinear)
        self.up4 = UpTransition(dim=dim, in_channels=out_shapes[1], out_channels=out_shapes[0], bilinear=bilinear)

        self.out_tr = Conv(dim=dim, in_channels=out_shapes[0], out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_tr(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out_tr(x)

        return out

    def calc_loss(self, x):
        pass
