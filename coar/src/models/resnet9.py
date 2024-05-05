import torch as ch

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

def ResNet9(num_classes, channels_last=True):
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )

    if channels_last:
        model = model.to(memory_format=ch.channels_last)

    return model

def ResNet9Mul(num_classes, channels_last=True, mul=1):
    nc = lambda num_conv: int(num_conv*mul)
    model = ch.nn.Sequential(
        conv_bn(3, nc(64), kernel_size=3, stride=1, padding=1),
        conv_bn(nc(64), nc(128), kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(nc(128), nc(128)), conv_bn(nc(128), nc(128)))),
        conv_bn(nc(128), nc(256), kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(nc(256), nc(256)), conv_bn(nc(256), nc(256)))),
        conv_bn(nc(256), nc(128), kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(nc(128), num_classes, bias=False),
        Mul(0.2)
    )

    if channels_last:
        model = model.to(memory_format=ch.channels_last)

    return model