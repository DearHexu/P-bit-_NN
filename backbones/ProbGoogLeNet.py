'''GoogLeNet with PyTorch. Includes original GoogLeNet (ReLU) and probabilistic version ProbGoogLeNet (BinarySTE + configurable activation).'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.BinarySTE import BinarySTE


# ---------- Original Inception (ReLU, no BinarySTE), for GoogLeNet ----------
class InceptionBase(nn.Module):
    """Original Inception: Conv+BN+ReLU, no BinarySTE."""
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(InceptionBase, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    """Original GoogLeNet (ReLU), for CIFAR10/CIFAR100 non-probabilistic baselines."""
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.a3 = InceptionBase(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBase(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = InceptionBase(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionBase(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionBase(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionBase(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionBase(528, 256, 160, 320, 32, 128, 128)  # 528 = d4 output channels
        self.a5 = InceptionBase(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBase(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class TanhSigmoid(nn.Module):
    def __init__(self):
        super(TanhSigmoid, self).__init__()
    def forward(self, x):
        return F.tanh(x) * F.sigmoid(x)


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, activation="tanh_sigmoid"):
        super(Inception, self).__init__()
        from core.activations import get_activation
        act = get_activation(activation or "tanh_sigmoid")
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            act,
            BinarySTE(mode='uniform'),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            act,
            BinarySTE(mode='uniform'),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            get_activation(activation or "tanh_sigmoid"),
            BinarySTE(mode='uniform'),
        )
        act2 = get_activation(activation or "tanh_sigmoid")
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            act2,
            BinarySTE(mode='uniform'),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            get_activation(activation or "tanh_sigmoid"),
            BinarySTE(mode='uniform'),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            get_activation(activation or "tanh_sigmoid"),
            BinarySTE(mode='uniform'),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            get_activation(activation or "tanh_sigmoid"),
            BinarySTE(mode='uniform'),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class ProbGoogLeNet(nn.Module):
    def __init__(self, num_classes=10, activation="tanh_sigmoid"):
        super(ProbGoogLeNet, self).__init__()
        from core.activations import get_activation
        act = get_activation(activation or "tanh_sigmoid")
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            act,
        )
        self.binary1 = BinarySTE(mode="uniform")
        act_name = activation or "tanh_sigmoid"
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, activation=act_name)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, activation=act_name)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, activation=act_name)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, activation=act_name)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, activation=act_name)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, activation=act_name)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, activation=act_name)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, activation=act_name)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, activation=act_name)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.binary1(out)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
