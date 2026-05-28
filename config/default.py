# -*- coding: utf-8 -*-
"""
默认配置与实验配置定义。
用于概率计算网络（PbitNet）在多数据集上的训练与对比实验（含 tanh*sigmoid 复合激活）。
"""

from dataclasses import dataclass
from typing import Optional, List
import argparse

from config.batch_run_config import MODEL_CHOICES, DATASET_CHOICES, ACTIVATION_CHOICES

# 各数据集默认下载/存放根目录
DATASET_DEFAULT_ROOT: dict = {
    "CIFAR10": "CIFAR10", "CIFAR100": "CIFAR100", "MNIST": "data", "FashionMNIST": "fashion",
}


@dataclass
class ExperimentConfig:
    """单次实验的完整配置（便于多实验并行时按配置区分）。"""

    # 实验标识（用于日志与保存路径）
    exp_name: str = "ProbGoogLeNet_CIFAR10"
    # 模型选择（见 MODEL_CHOICES）
    model_name: str = "ProbGoogLeNet"
    # 数据集: CIFAR10, CIFAR100, MNIST, FashionMNIST
    dataset: str = "CIFAR10"
    data_root: str = ""
    num_classes: int = 10
    # 训练
    epochs: int = 100
    batch_size: int = 128
    test_batch_size: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # 学习率调度（与 epochs 一致即可）
    T_max: int = 100
    # 优化器: AdamW, Adam, SGD
    optimizer: str = "AdamW"
    # 损失: CrossEntropy, BCEWithLogits
    criterion: str = "CrossEntropy"
    # 数据加载
    num_workers: int = 2
    # 恢复训练
    resume: bool = False
    checkpoint_path: Optional[str] = None
    # 输出根目录（所有实验的父目录）
    output_root: str = "outputs"
    # 是否使用 DataParallel（多卡同模型）
    use_data_parallel: bool = False
    # 当前进程使用的 GPU ID（多进程并行时由 run_parallel 设置）
    gpu_id: Optional[int] = None
    # 激活函数（见 ACTIVATION_CHOICES），仅对 Prob* 等支持激活配置的模型生效
    activation: str = "tanh_sigmoid"

    def __post_init__(self):
        if self.T_max <= 0:
            self.T_max = self.epochs
        if self.checkpoint_path is None:
            self.checkpoint_path = f"checkpoint/{self.model_name}-{self.dataset}.pth"
        if not self.data_root:
            self.data_root = DATASET_DEFAULT_ROOT.get(self.dataset, self.dataset)


def get_config(args: Optional[argparse.Namespace] = None) -> ExperimentConfig:
    """
    从命令行参数构建配置；未传入则返回默认配置。
    命令行可覆盖 exp_name, model_name, epochs, batch_size, lr, resume, checkpoint 等。
    """
    cfg = ExperimentConfig()
    if args is None:
        return cfg
    if getattr(args, "exp_name", None) is not None:
        cfg.exp_name = args.exp_name
    if getattr(args, "model_name", None) is not None:
        cfg.model_name = args.model_name
    if getattr(args, "dataset", None) is not None:
        cfg.dataset = args.dataset
        cfg.data_root = DATASET_DEFAULT_ROOT.get(cfg.dataset, cfg.data_root or cfg.dataset)
    if getattr(args, "data", None) is not None:
        cfg.data_root = args.data
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
        cfg.T_max = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "lr", None) is not None:
        cfg.lr = args.lr
    if getattr(args, "resume", None) is not None and args.resume:
        cfg.resume = True
    if getattr(args, "checkpoint", None) is not None:
        cfg.checkpoint_path = args.checkpoint
    if getattr(args, "output_root", None) is not None:
        cfg.output_root = args.output_root
    if getattr(args, "gpu_id", None) is not None:
        cfg.gpu_id = args.gpu_id
    if getattr(args, "activation", None) is not None:
        cfg.activation = args.activation
    return cfg


def parse_args() -> argparse.Namespace:
    """解析命令行参数，供单机训练脚本使用。"""
    parser = argparse.ArgumentParser(description="PbitNet CIFAR10 训练（tanh*sigmoid 概率计算网络）")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称，用于日志与保存路径")
    parser.add_argument("--model_name", type=str, default="ProbGoogLeNet", choices=MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=DATASET_CHOICES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data", type=str, default="", help="数据根目录，空则用 dataset 默认")
    parser.add_argument("--T_max", type=int, default=100, help="CosineAnnealingLR 周期")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--resume", "-r", action="store_true", help="从 checkpoint 恢复")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--output_root", type=str, default="outputs", help="实验结果根目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用的 GPU ID")
    parser.add_argument("--activation", type=str, default="tanh_sigmoid", choices=ACTIVATION_CHOICES,
                        help="激活函数（仅对 Prob* 等模型生效）")
    return parser.parse_args()
