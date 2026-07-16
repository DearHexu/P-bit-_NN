"""
PyTorch implementation of ResNet18/34/50/101/152. Includes original ResNet (ReLU) and probabilistic version ProbResNet (BinarySTE + configurable activation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.BinarySTE import BinarySTE


# ---------- Original ResNet building blocks (ReLU, no BinarySTE), for ResNet18/34/50/101/152 ----------
class BasicBlockBase(nn.Module):
    """Original BasicBlock: Conv+BN+ReLU, no BinarySTE."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockBase, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckBase(nn.Module):
    """Original Bottleneck: Conv+BN+ReLU, no BinarySTE."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBase, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """Original ResNet (ReLU), for CIFAR10/CIFAR100 non-probabilistic baselines."""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def ResNet18(num_classes=10):
    return ResNet(BasicBlockBase, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlockBase, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(BottleneckBase, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(BottleneckBase, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(BottleneckBase, [3, 8, 36, 3], num_classes=num_classes)


# ---------- Probabilistic ResNet (BinarySTE + configurable activation) ----------
# Build ResNet18-34 basic blocks (activation configurable, for comparison experiments)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=None):
        super(BasicBlock, self).__init__()
        from core.activations import get_activation
        self.act = get_activation(activation if activation else "tanh_sigmoid")
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary1 = BinarySTE(mode="uniform")
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary2 = BinarySTE(mode="uniform")
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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


# Build ResNet50/101/152 Bottleneck blocks (activation configurable)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation=None):
        super(Bottleneck, self).__init__()
        from core.activations import get_activation
        self.act = get_activation(activation if activation else "tanh_sigmoid")
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.binary1 = BinarySTE(mode="uniform")
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binary2 = BinarySTE(mode="uniform")
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.binary3 = BinarySTE(mode="uniform")
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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


# Build ProbResNet blocks (supports activation config, for comparison against relu/silu etc.)
class ProbResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation="tanh_sigmoid"):
        super(ProbResNet, self).__init__()
        from core.activations import get_activation
        self._activation = activation or "tanh_sigmoid"
        self.act = get_activation(self._activation)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.binary1 = BinarySTE(mode="uniform")
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Xavier init
        # Core idea: keep variance consistent between input and output
        # This prevents all output values from collapsing to 0
        # This is a general method applicable to any activation function
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

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
        out = self.linear(out)
        return out

def ProbResNet18(num_classes=10, activation="tanh_sigmoid"):
    return ProbResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, activation=activation)


def ProbResNet34(num_classes=10, activation="tanh_sigmoid"):
    return ProbResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, activation=activation)


def ProbResNet50(num_classes=10, activation="tanh_sigmoid"):
    return ProbResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, activation=activation)


def ProbResNet101(num_classes=10, activation="tanh_sigmoid"):
    return ProbResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, activation=activation)


def ProbResNet152(num_classes=10, activation="tanh_sigmoid"):
    return ProbResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, activation=activation)

# Test
if __name__ == '__main__':
    model = ProbResNet18()
    print(model)

    input = torch.randn(1, 3, 32, 32)
    out = model(input)
    print(out.shape)
