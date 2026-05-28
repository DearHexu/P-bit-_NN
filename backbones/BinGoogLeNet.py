# -*- coding: utf-8 -*-
"""
全二值权重 GoogLeNet（BinGoogLeNet）：权重仅 ±1，无缩放，STE 反向；激活与配置方式同 ProbGoogLeNet。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.BinarySTE import BinarySTE
from backbones.BinaryWeight import Conv2dBin, LinearBin


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, activation="tanh_sigmoid"):
        super().__init__()
        from core.activations import get_activation
        act = get_activation(activation or "tanh_sigmoid")
        self.b1 = nn.Sequential(
            Conv2dBin(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            act,
        )
        self.b2 = nn.Sequential(
            Conv2dBin(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            act,
            Conv2dBin(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            get_activation(activation or "tanh_sigmoid"),
        )
        act2 = get_activation(activation or "tanh_sigmoid")
        self.b3 = nn.Sequential(
            Conv2dBin(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            act2,
            Conv2dBin(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            get_activation(activation or "tanh_sigmoid"),
            Conv2dBin(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            get_activation(activation or "tanh_sigmoid"),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            Conv2dBin(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            get_activation(activation or "tanh_sigmoid"),
        )
        self.binary1 = BinarySTE(mode="uniform")
        self.binary2 = BinarySTE(mode="uniform")
        self.binary3 = BinarySTE(mode="uniform")
        self.binary4 = BinarySTE(mode="uniform")

    def forward(self, x):
        y1 = self.b1(x)
        y1 = self.binary1(y1)
        y2 = self.b2(x)
        y2 = self.binary2(y2)
        y3 = self.b3(x)
        y3 = self.binary3(y3)
        y4 = self.b4(x)
        y4 = self.binary4(y4)
        return torch.cat([y1, y2, y3, y4], 1)


class BinGoogLeNet(nn.Module):
    """全二值权重 GoogLeNet：所有 Conv/Linear 为 ±1 二值，激活可配置。"""

    def __init__(self, num_classes=10, activation="tanh_sigmoid"):
        super().__init__()
        from core.activations import get_activation
        act = get_activation(activation or "tanh_sigmoid")
        self.pre_layers = nn.Sequential(
            Conv2dBin(3, 192, kernel_size=3, padding=1),
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
        self.linear = LinearBin(1024, num_classes)

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
        return self.linear(out)


if __name__ == "__main__":
    model = BinGoogLeNet(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
