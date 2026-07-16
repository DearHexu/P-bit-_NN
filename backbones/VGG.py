'''VGG11/13/16/19 in Pytorch. Includes original and probabilistic version ProbVGG (BinarySTE + configurable activation).'''
import torch
import torch.nn as nn
from backbones.BinarySTE import BinarySTE

# M denotes a max pooling layer
cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """Original VGG (Conv+BN+ReLU)."""
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_prob=False, activation=None)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, cfg_list, use_prob=False, activation=None):
        layers = []
        in_channels = 3
        for x in cfg_list:
            if x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if use_prob and activation is not None:
                    from core.activations import get_activation
                    act = get_activation(activation or "tanh_sigmoid")
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        act,
                        BinarySTE(mode="uniform"),
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ))
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class ProbVGG(nn.Module):
    """Probabilistic VGG: Conv+BN+Activation+BinarySTE per layer, activation configurable."""
    def __init__(self, vgg_name, num_classes=10, activation="tanh_sigmoid"):
        super(ProbVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_prob=True, activation=activation or "tanh_sigmoid")
        from core.activations import get_activation
        act = get_activation(activation or "tanh_sigmoid")
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            act,
            BinarySTE(mode="uniform"),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, cfg_list, use_prob=False, activation=None):
        layers = []
        in_channels = 3
        for x in cfg_list:
            if x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if use_prob and activation is not None:
                    from core.activations import get_activation
                    act = get_activation(activation or "tanh_sigmoid")
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        act,
                        BinarySTE(mode="uniform"),
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ))
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


def VGG11(num_classes=10):
    return VGG("VGG11", num_classes=num_classes)


def VGG13(num_classes=10):
    return VGG("VGG13", num_classes=num_classes)


def VGG16(num_classes=10):
    return VGG("VGG16", num_classes=num_classes)


def VGG19(num_classes=10):
    return VGG("VGG19", num_classes=num_classes)
