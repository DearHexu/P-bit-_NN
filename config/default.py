# -*- coding: utf-8 -*-
"""
Default config and experiment config definition.
For probabilistic computing network (PbitNet) training and comparison experiments across multiple datasets (including tanh*sigmoid composite activation).
"""

from dataclasses import dataclass
from typing import Optional, List
import argparse

from config.batch_run_config import MODEL_CHOICES, DATASET_CHOICES, ACTIVATION_CHOICES

# Default download/storage root per dataset
DATASET_DEFAULT_ROOT: dict = {
    "CIFAR10": "CIFAR10", "CIFAR100": "CIFAR100", "MNIST": "data", "FashionMNIST": "fashion",
}


@dataclass
class ExperimentConfig:
    """Complete config for a single experiment (convenient for distinguishing across parallel runs)."""

    # Experiment identifier (used for logging and save paths)
    exp_name: str = "ProbGoogLeNet_CIFAR10"
    # Model selection (see MODEL_CHOICES)
    model_name: str = "ProbGoogLeNet"
    # Dataset: CIFAR10, CIFAR100, MNIST, FashionMNIST
    dataset: str = "CIFAR10"
    data_root: str = ""
    num_classes: int = 10
    # Training
    epochs: int = 100
    batch_size: int = 128
    test_batch_size: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # LR schedule (match to epochs)
    T_max: int = 100
    # Optimizer: AdamW, Adam, SGD
    optimizer: str = "AdamW"
    # Loss: CrossEntropy, BCEWithLogits
    criterion: str = "CrossEntropy"
    # Data loading
    num_workers: int = 2
    # Resume training
    resume: bool = False
    checkpoint_path: Optional[str] = None
    # Output root directory (parent of all experiments)
    output_root: str = "outputs"
    # Whether to use DataParallel (multiple GPUs, same model)
    use_data_parallel: bool = False
    # GPU ID for the current process (set by run_parallel during multi-process runs)
    gpu_id: Optional[int] = None
    # Activation function (see ACTIVATION_CHOICES), only effective for Prob* models that support activation config
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
    Build config from CLI args; returns default config if none provided.
    CLI can override exp_name, model_name, epochs, batch_size, lr, resume, checkpoint, etc.
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
    """Parse CLI arguments for single-machine training scripts."""
    parser = argparse.ArgumentParser(description="PbitNet CIFAR10 training (tanh*sigmoid probabilistic computing network)")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name, used for logging and save paths")
    parser.add_argument("--model_name", type=str, default="ProbGoogLeNet", choices=MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=DATASET_CHOICES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data", type=str, default="", help="Data root directory; uses dataset default if empty")
    parser.add_argument("--T_max", type=int, default=100, help="CosineAnnealingLR period")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--output_root", type=str, default="outputs", help="Experiment results root directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--activation", type=str, default="tanh_sigmoid", choices=ACTIVATION_CHOICES,
                        help="Activation function (only effective for Prob* models)")
    return parser.parse_args()
