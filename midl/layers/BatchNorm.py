"""
A wrapper for batch normalization layer

Author      : Sanguk Park
Version     : 0.1
"""

import torch.nn as nn


class BN(nn.Module):
    def __init__(
            self,
            dim: int,
            channels: int,
    ):
        super(BN, self).__init__()
        
        if dim == 2:
            self.bn = nn.BatchNorm2d(channels)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(channels)
        else:
            raise ValueError("Only 2 or 3 is supported for 'dim' in {}".format(self.__class__.__name__))

    def forward(self, x):
        return self.bn(x)
