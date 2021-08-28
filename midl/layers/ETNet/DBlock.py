import torch.nn as nn
from ..Conv import Conv


class DBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int):
        super(DBlock, self).__init__()

        self.conv1 = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=False
        )
        if dim == 2:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif dim == 3:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            bias=False
        )
        self.conv3 = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=False,
            groups=out_channels
        )
        self.conv4 = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + skip
