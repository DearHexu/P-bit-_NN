# -*- coding: utf-8 -*-
"""
Binary-weight layers: forward uses ±1 only, no scaling, backward STE.
Used by BinResNet / BinGoogLeNet and other fully binary-weight networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sign_ste(w: torch.Tensor) -> torch.Tensor:
    """Binarize weights to ±1, straight-through estimator backward. w is the full-precision weight."""
    # Only ±1: w>=0 -> 1, w<0 -> -1 (consistent with sign, avoids 0)
    w_b = 2.0 * (w >= 0).to(w.dtype) - 1.0
    return w_b.detach() + w - w.detach()


class Conv2dBin(nn.Module):
    """Binary-weight Conv2d: forward uses sign(W)∈{±1}, backward STE updates full-precision W."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w_ste = _sign_ste(self.weight)
        return F.conv2d(x, w_ste, self.bias, self.stride, self.padding)


class LinearBin(nn.Module):
    """Binary-weight Linear: forward uses sign(W)∈{±1}, backward STE updates full-precision W."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w_ste = _sign_ste(self.weight)
        return F.linear(x, w_ste, self.bias)
