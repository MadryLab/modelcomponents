import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .cifar_resnet import BasicBlock

def recursive_getattr(obj, attr, *args):    
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class MiniResNet(nn.Module):

    KERNEL_SIZE = 3
    IN_CHANNELS = 3
    AVG_POOL_SIZE = 8
    LINEAR_MUL = 4*4

    def __init__(self, num_blocks=4, in_planes=128, num_classes=10, debug=False):
        super().__init__()
        self.num_classes = num_classes
        self.debug = debug
        self.in_planes = in_planes
        self.kernel_size = self.KERNEL_SIZE
        self.num_blocks = num_blocks

        # pre-layer stuff
        self.conv1 = nn.Conv2d(self.IN_CHANNELS, self.in_planes, kernel_size=self.kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # make single layer with K BasicBlocks
        # BasicBLock: conv1, bn1, conv2, bn2, shortcut
        # each conv has `in_planes` filters
        get_block = lambda: BasicBlock(self.in_planes, self.in_planes, stride=1)
        self.layer = nn.Sequential(*[get_block() for _ in range(num_blocks)])

        # register blocks with setattr to make it compatible with masking code
        for idx, block in enumerate(self.layer):
            setattr(self, f'block{idx}', block)
            
        # post-layer stuff
        self.flatten = nn.Flatten()
        self.avg_pool_2d = nn.AvgPool2d(self.AVG_POOL_SIZE)
        self.linear = nn.Linear(self.in_planes*self.LINEAR_MUL, num_classes)

    def forward(self, x, *args, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = self.avg_pool_2d(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out