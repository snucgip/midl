"""
An implementation for PsiNet
https://arxiv.org/abs/1902.04099

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn

from midl.layers.Conv import Conv
from midl.layers.ConvBlock import ConvBlock
from midl.layers.UNet.DownTransition import DownTransition
from midl.layers.UNet.UpTransition import UpTransition


class PsiNet(nn.Module):
    def __init__(
            self,
            dim,
            in_channels,
            n_classes,
            bilinear=True
    ):
        super(PsiNet, self).__init__()

        if dim == 2:
            out_shapes = [64, 128, 256, 512, 1024]
        elif dim == 3:
            out_shapes = [16, 32, 64, 128, 256]

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

        # Mask decoder
        self.up1 = UpTransition(dim=dim, in_channels=out_shapes[4], out_channels=out_shapes[3]//factor, bilinear=bilinear)
        self.up2 = UpTransition(dim=dim, in_channels=out_shapes[3], out_channels=out_shapes[2]//factor, bilinear=bilinear)
        self.up3 = UpTransition(dim=dim, in_channels=out_shapes[2], out_channels=out_shapes[1]//factor, bilinear=bilinear)
        self.up4 = UpTransition(dim=dim, in_channels=out_shapes[1], out_channels=out_shapes[0], bilinear=bilinear)
        self.out_tr = Conv(dim=dim, in_channels=out_shapes[0], out_channels=n_classes, kernel_size=1)

        # Contour decoder
        self.c_up1 = UpTransition(dim=dim, in_channels=out_shapes[4], out_channels=out_shapes[3] // factor,
                                  bilinear=bilinear)
        self.c_up2 = UpTransition(dim=dim, in_channels=out_shapes[3], out_channels=out_shapes[2] // factor,
                                  bilinear=bilinear)
        self.c_up3 = UpTransition(dim=dim, in_channels=out_shapes[2], out_channels=out_shapes[1] // factor,
                                  bilinear=bilinear)
        self.c_up4 = UpTransition(dim=dim, in_channels=out_shapes[1], out_channels=out_shapes[0], bilinear=bilinear)
        self.c_out_tr = Conv(dim=dim, in_channels=out_shapes[0], out_channels=n_classes, kernel_size=1)

        # Distance Map decoder
        self.dt_up1 = UpTransition(dim=dim, in_channels=out_shapes[4], out_channels=out_shapes[3] // factor,
                                   bilinear=bilinear)
        self.dt_up2 = UpTransition(dim=dim, in_channels=out_shapes[3], out_channels=out_shapes[2] // factor,
                                   bilinear=bilinear)
        self.dt_up3 = UpTransition(dim=dim, in_channels=out_shapes[2], out_channels=out_shapes[1] // factor,
                                   bilinear=bilinear)
        self.dt_up4 = UpTransition(dim=dim, in_channels=out_shapes[1], out_channels=out_shapes[0], bilinear=bilinear)
        self.dt_out_tr = Conv(dim=dim, in_channels=out_shapes[0], out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_tr(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        mask_x = self.up1(x5, x4)
        mask_x = self.up2(mask_x, x3)
        mask_x = self.up3(mask_x, x2)
        mask_x = self.up4(mask_x, x1)
        mask_out = self.out_tr(mask_x)

        c_x = self.c_up1(x5, x4)
        c_x = self.c_up2(c_x, x3)
        c_x = self.c_up3(c_x, x2)
        c_x = self.c_up4(c_x, x1)
        c_out = self.c_out_tr(c_x)

        dt_x = self.dt_up1(x5, x4)
        dt_x = self.dt_up2(dt_x, x3)
        dt_x = self.dt_up3(dt_x, x2)
        dt_x = self.dt_up4(dt_x, x1)
        dt_out = self.dt_out_tr(dt_x)

        return mask_out, c_out, dt_out

    def calc_loss(self):
        pass
