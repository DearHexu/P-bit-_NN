"""
绘制各种激活函数的图像
包括: Sigmoid, ReLU, SiLU (Swish), Tanh, Leaky ReLU, GELU 等
"""

import numpy as np
# import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 支持中文显示
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)


def silu(x):
    """SiLU (Swish): x * sigmoid(x)"""
    return x * sigmoid(x)

def tanhsig(x):
    """tanhsig: tanh(x) * sigmoid(x)"""
    return tanh(x) * sigmoid(x) 


def tanh(x):
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: x if x > 0 else alpha * x"""
    return np.where(x > 0, x, alpha * x)


def gelu(x):
    """GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def softplus(x):
    """Softplus: log(1 + exp(x))"""
    return np.log(1 + np.exp(np.clip(x, -500, 500)))

def mish(x):
    """Mish: x * tanh(softplus(x))"""
    return x * np.tanh(softplus(x))

def gumbel_sigmoid(logits, temperature=1.0, hard=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    eps = 1e-20  # 防止 log(0)
    uniform_noise = rng.uniform(0, 1, size=logits.shape)
    gumbels = -np.log(-np.log(uniform_noise + eps) + eps)
    y_soft = 1.0 / (1.0 + np.exp(-(logits + gumbels) / temperature))
    
    if not hard:
        return y_soft
    y_hard = (y_soft > 0.5).astype(np.float32)

    return y_hard  # 前向值就是硬采样结果


def plot_activation(ax, x, y, name, color='#2E86AB'):
    """在子图上绘制单个激活函数"""
    ax.plot(x, y, color=color, linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(name, fontsize=12)
    ax.set_xlim(x.min(), x.max())
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')


def main():
    x = np.linspace(-50, 50, 10000)

    activations = [
        (sigmoid(x), 'Sigmoid', '#E94F37'),
        (relu(x), 'ReLU', '#2E86AB'),
        (tanh(x), 'Tanh', '#F18F01'),
        (silu(x), 'SiLU (Swish)', '#44AF69'),
        (gelu(x), 'GELU', '#6C5CE7'),
        # (mish(x), 'Mish', '#C73E1D'),
        (tanhsig(x), 'Tanhsig', '#00B894'),
    ]

    n = len(activations)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, (y, name, color) in enumerate(activations):
        row, col = idx // n_cols, idx % n_cols
        plot_activation(axes[row, col], x, y, name, color)

    # 隐藏多余的子图
    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('activation functions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    print('图像已保存为 activation_functions.png')
    plt.show()


if __name__ == '__main__':
    main()
