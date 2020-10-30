"""
A wrapper for convolutional layer

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn

class Conv(nn.Module):
    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'
    ):
        # Check parameters
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size, ) * dim
        else:
            assert len(kernel_size) == dim

        if type(stride) != tuple:
            stride = (stride, ) * dim
        else:
            assert len(stride) == dim

        if type(padding) != tuple:
            padding = (padding, ) * dim
        else:
            assert len(padding) == dim

        if type(dilation) != tuple:
            dilation = (dilation, ) * dim
        else:
            assert len(dilation) == dim

        super(Conv, self).__init__()

        if dim == 2:
            self.conv = nn.Conv2d(
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
        elif dim == 3:
            self.conv = nn.Conv3d(
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
        else:
            raise ValueError("Only 2 or 3 is supported for 'dim' in {}".format(self.__class__.__name__))

    def forward(self, x):
        return self.conv(x)


class ConvTranspose(nn.Module):
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
        # Check parameters
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size,) * dim
        else:
            assert len(kernel_size) == dim

        if type(stride) != tuple:
            stride = (stride,) * dim
        else:
            assert len(stride) == dim

        if type(padding) != tuple:
            padding = (padding,) * dim
        else:
            assert len(padding) == dim

        if type(dilation) != tuple:
            dilation = (dilation,) * dim
        else:
            assert len(dilation) == dim

        super(ConvTranspose, self).__init__()

        if dim == 2:
            self.conv = nn.ConvTranspose2d(
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
        elif dim == 3:
            self.conv = nn.ConvTranspose3d(
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
        else:
            raise ValueError("Only 2 or 3 is supported for 'dim' in {}".format(self.__class__.__name__))

    def forward(self, x):
        return self.conv(x)