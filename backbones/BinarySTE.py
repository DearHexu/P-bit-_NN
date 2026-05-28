import torch
import torch.nn as nn


class BinarySTE(nn.Module):
    def __init__(self, mode='uniform', threshold_std=0.2):
        super(BinarySTE, self).__init__()
        # mode: 'uniform' 均匀采样 [0,1)；'gaussian' 均值 0.5 的高斯采样（裁剪到 [0,1]）
        self.mode = mode
        self.threshold_std = threshold_std

    def forward(self, x):
        if self.mode == 'uniform':
            threshold = torch.rand_like(x, device=x.device, dtype=x.dtype)
        else:
            # gaussian: N(0.5, std^2)，裁剪到 [0, 1]
            threshold = torch.clamp(0.5 + self.threshold_std * torch.randn_like(x, device=x.device, dtype=x.dtype), 0.0, 1.0)
        binary_out = (x > threshold).float()
        return binary_out.detach() + x - x.detach()

