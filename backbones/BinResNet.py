# -*- coding: utf-8 -*-
"""
全二值权重 ResNet（BinResNet）：权重仅 ±1，无缩放，STE 反向；激活与配置方式同 ProbResNet。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.BinarySTE import BinarySTE
from backbones.BinaryWeight import Conv2dBin, LinearBin


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=None):
        super().__init__()
        from core.activations import get_activation
        self.act = get_activation(activation or "tanh_sigmoid")
        self.conv1 = Conv2dBin(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary1 = BinarySTE(mode="uniform")
        self.conv2 = Conv2dBin(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary2 = BinarySTE(mode="uniform")
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2dBin(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.binary1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        out = self.binary2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation=None):
        super().__init__()
        from core.activations import get_activation
        self.act = get_activation(activation or "tanh_sigmoid")
        self.conv1 = Conv2dBin(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary1 = BinarySTE(mode="uniform")
        self.conv2 = Conv2dBin(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary2 = BinarySTE(mode="uniform")
        self.conv3 = Conv2dBin(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.binary3 = BinarySTE(mode="uniform")
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2dBin(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.binary1(out)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.binary2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act(out)
        out = self.binary3(out)
        return out


class BinResNet(nn.Module):
    """全二值权重 ResNet：所有 Conv/Linear 为 ±1 二值，激活可配置，不做 weight decay、学习率可调大。"""

    def __init__(self, block, num_blocks, num_classes=10, activation="tanh_sigmoid"):
        super().__init__()
        from core.activations import get_activation
        self._activation = activation or "tanh_sigmoid"
        self.act = get_activation(self._activation)
        self.in_planes = 64
        self.conv1 = Conv2dBin(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.binary1 = BinarySTE(mode="uniform")
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = LinearBin(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        act_name = getattr(self, "_activation", "tanh_sigmoid")
        for s in strides:
            layers.append(block(self.in_planes, planes, s, activation=act_name))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.binary1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def BinResNet18(num_classes=10, activation="tanh_sigmoid"):
    return BinResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, activation=activation)


def BinResNet34(num_classes=10, activation="tanh_sigmoid"):
    return BinResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, activation=activation)


def BinResNet50(num_classes=10, activation="tanh_sigmoid"):
    return BinResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, activation=activation)


def BinResNet101(num_classes=10, activation="tanh_sigmoid"):
    return BinResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, activation=activation)


def BinResNet152(num_classes=10, activation="tanh_sigmoid"):
    return BinResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, activation=activation)


if __name__ == "__main__":
    model = BinResNet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
