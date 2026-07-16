# -*- coding: utf-8 -*-
"""
Configurable activation functions for ProbResNet / ProbGoogLeNet etc. comparison experiments.
Corresponds to ACTIVATION_CHOICES in config; obtain nn.Module via get_activation(name).
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# Must match config.batch_run_config.ACTIVATION_CHOICES
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(inplace=True),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "tanh_sigmoid": None,  # Custom, see below
    "silu": nn.SiLU(inplace=True),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
}


class TanhSigmoid(nn.Module):
    """Composite activation: tanh(x) * sigmoid(x), used for probabilistic computing network comparison experiments."""
    def forward(self, x):
        return F.tanh(x) * F.sigmoid(x)


def get_activation(name: str) -> nn.Module:
    """
    Return a new activation function module instance by config name (avoids sharing the same module across models).
    For valid name values see config.batch_run_config.ACTIVATION_CHOICES.
    """
    name = (name or "tanh_sigmoid").strip().lower()
    if name == "tanh_sigmoid":
        return TanhSigmoid()
    if name in ACTIVATION_REGISTRY and ACTIVATION_REGISTRY[name] is not None:
        # Return new instance (for ReLU/SiLU etc. that can be copied)
        m = ACTIVATION_REGISTRY[name]
        if isinstance(m, nn.ReLU):
            return nn.ReLU(inplace=True)
        if isinstance(m, nn.SiLU):
            return nn.SiLU(inplace=True)
        if isinstance(m, nn.LeakyReLU):
            return nn.LeakyReLU(0.01, inplace=True)
        if isinstance(m, (nn.Sigmoid, nn.Tanh, nn.GELU)):
            return type(m)()
    raise ValueError(f"Unknown activation: {name}, choices: relu, sigmoid, tanh, tanh_sigmoid, silu, gelu, leaky_relu")
