'''LeNet in PyTorch. 含原版与概率网络化版本 ProbLeNet5/ProbLeNet3（BinarySTE + 可配置激活）。'''
import torch.nn as nn
import torch.nn.functional as F
from backbones.BinarySTE import BinarySTE


class LeNet5(nn.Module):
    """原版 LeNet5。"""
    def __init__(self, num_class=10, in_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, num_class)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn4(self.fc1(out)))
        return self.fc2(out)


class ProbLeNet5(nn.Module):
    """概率网络化 LeNet5：Conv+BN+激活+BinarySTE，激活可配置。"""
    def __init__(self, num_class=10, in_channels=1, activation="tanh_sigmoid"):
        super(ProbLeNet5, self).__init__()
        from core.activations import get_activation
        self.act = get_activation(activation or "tanh_sigmoid")
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.binary1 = BinarySTE(mode="uniform")
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.binary2 = BinarySTE(mode="uniform")
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.bn3 = nn.BatchNorm2d(120)
        self.binary3 = BinarySTE(mode="uniform")
        self.fc1 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, num_class)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.binary1(out)
        out = F.max_pool2d(out, 2)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.binary2(out)
        out = F.max_pool2d(out, 2)
        out = self.act(self.bn3(self.conv3(out)))
        out = self.binary3(out)
        out = out.view(out.size(0), -1)
        out = self.act(self.bn4(self.fc1(out)))
        return self.fc2(out)


class LeNet3(nn.Module):
    """原版 LeNet3。"""
    def __init__(self, num_class=10, in_channels=1):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, num_class)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn4(self.fc1(out)))
        return self.fc2(out)


class ProbLeNet3(nn.Module):
    """概率网络化 LeNet3：Conv+BN+激活+BinarySTE，激活可配置。"""
    def __init__(self, num_class=10, in_channels=1, activation="tanh_sigmoid"):
        super(ProbLeNet3, self).__init__()
        from core.activations import get_activation
        self.act = get_activation(activation or "tanh_sigmoid")
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.binary1 = BinarySTE(mode="uniform")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.binary2 = BinarySTE(mode="uniform")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.binary3 = BinarySTE(mode="uniform")
        self.fc1 = nn.Linear(128, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, num_class)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.binary1(out)
        out = F.max_pool2d(out, 2)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.binary2(out)
        out = F.max_pool2d(out, 2)
        out = self.act(self.bn3(self.conv3(out)))
        out = self.binary3(out)
        out = out.view(out.size(0), -1)
        out = self.act(self.bn4(self.fc1(out)))
        return self.fc2(out)
