# -*- coding: utf-8 -*-
"""
可配置激活函数：供 ProbResNet / ProbGoogLeNet 等做对比实验。
与 config 中的 ACTIVATION_CHOICES 对应，通过 get_activation(name) 获取 nn.Module。
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# 与 config.batch_run_config.ACTIVATION_CHOICES 保持一致
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(inplace=True),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "tanh_sigmoid": None,  # 自定义，见下
    "silu": nn.SiLU(inplace=True),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
}


class TanhSigmoid(nn.Module):
    """复合激活: tanh(x) * sigmoid(x)，用于概率计算网络对比实验。"""
    def forward(self, x):
        return F.tanh(x) * F.sigmoid(x)


def get_activation(name: str) -> nn.Module:
    """
    根据配置名返回激活函数模块（新建实例，避免多模型共享同一模块）。
    name 取值见 config.batch_run_config.ACTIVATION_CHOICES。
    """
    name = (name or "tanh_sigmoid").strip().lower()
    if name == "tanh_sigmoid":
        return TanhSigmoid()
    if name in ACTIVATION_REGISTRY and ACTIVATION_REGISTRY[name] is not None:
        # 返回新实例（ReLU/SiLU 等可复制的）
        m = ACTIVATION_REGISTRY[name]
        if isinstance(m, nn.ReLU):
            return nn.ReLU(inplace=True)
        if isinstance(m, nn.SiLU):
            return nn.SiLU(inplace=True)
        if isinstance(m, nn.LeakyReLU):
            return nn.LeakyReLU(0.01, inplace=True)
        if isinstance(m, (nn.Sigmoid, nn.Tanh, nn.GELU)):
            return type(m)()
    raise ValueError(f"未知激活: {name}，可选: relu, sigmoid, tanh, tanh_sigmoid, silu, gelu, leaky_relu")
