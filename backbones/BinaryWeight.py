# -*- coding: utf-8 -*-
"""
二值权重层：前向仅 ±1，无缩放，反向 STE。
供 BinResNet / BinGoogLeNet 等全二值权重网络使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sign_ste(w: torch.Tensor) -> torch.Tensor:
    """权重二值化为 ±1，反向直通估计。w 为全精度权重。"""
    # 仅 ±1：w>=0 -> 1, w<0 -> -1（与 sign 一致，避免 0）
    w_b = 2.0 * (w >= 0).to(w.dtype) - 1.0
    return w_b.detach() + w - w.detach()


class Conv2dBin(nn.Module):
    """二值权重的 Conv2d：前向用 sign(W)∈{±1}，反向 STE 更新全精度 W。"""

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
    """二值权重的 Linear：前向用 sign(W)∈{±1}，反向 STE 更新全精度 W。"""

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
