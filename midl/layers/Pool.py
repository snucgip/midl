"""
A wrapper for pooling layer

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn

class Pool(nn.Module):
    def __init__(
            self,
            dim: int,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            mode='max'
    ):
        # Check parameters
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size, ) * dim
        else:
            assert len(kernel_size) == dim

        if stride != None:
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

        super(Pool, self).__init__()

        if dim == 2:
            if mode == 'max':
                self.pool = nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                )
            else:
                raise ValueError("Mode {} not supported for {}".format(mode, self.__class__.__name__))
        elif dim == 3:
            if mode == 'max':
                self.pool = nn.MaxPool3d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                )
            else:
                raise ValueError("Mode {} not supported for {}".format(mode, self.__class__.__name__))
        else:
            raise ValueError("Only 2 or 3 is supported for 'dim' in {}".format(self.__class__.__name__))

    def forward(self, x):
        return self.pool(x)
