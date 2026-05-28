'''AlexNet in PyTorch. 含原版与概率网络化版本 ProbAlexNet（BinarySTE + 可配置激活）。'''
import torch.nn as nn
from backbones.BinarySTE import BinarySTE


class AlexNet(nn.Module):
    """原版 AlexNet（CIFAR 适配，最后用 AdaptiveAvgPool2d）。"""
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ProbAlexNet(nn.Module):
    """概率网络化 AlexNet：每层 Conv+BN+激活+BinarySTE，激活可配置（与 ProbResNet 一致）。"""
    def __init__(self, num_classes=10, activation="tanh_sigmoid"):
        super(ProbAlexNet, self).__init__()
        from core.activations import get_activation
        act = get_activation(activation or "tanh_sigmoid")
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.act = act
        self.binary1 = BinarySTE(mode="uniform")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(192)
        self.binary2 = BinarySTE(mode="uniform")
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)
        self.binary3 = BinarySTE(mode="uniform")
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.binary4 = BinarySTE(mode="uniform")
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.binary5 = BinarySTE(mode="uniform")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 4096),
            get_activation(activation or "tanh_sigmoid"),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            get_activation(activation or "tanh_sigmoid"),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.binary1(out)
        out = self.pool(out)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.binary2(out)
        out = self.pool(out)
        out = self.act(self.bn3(self.conv3(out)))
        out = self.binary3(out)
        out = self.act(self.bn4(self.conv4(out)))
        out = self.binary4(out)
        out = self.act(self.bn5(self.conv5(out)))
        out = self.binary5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)
