# -*- coding: utf-8 -*-
"""
Multi-dataset support: CIFAR10, CIFAR100, MNIST, FashionMNIST.
Returns num_classes, data root, transforms and DataLoader based on dataset name.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
from config.default import ExperimentConfig


# Dataset metadata: (num_classes, image_size, channels, default data_root)
DATASET_META = {
    "CIFAR10": (10, 32, 3, "CIFAR10"),
    "CIFAR100": (100, 32, 3, "CIFAR100"),
    "MNIST": (10, 28, 1, "data"),
    "FashionMNIST": (10, 28, 1, "fashion"),
}


def get_dataset_meta(dataset_name: str) -> Tuple[int, int, int, str]:
    """Return (num_classes, size, in_channels, default_root)."""
    if dataset_name not in DATASET_META:
        raise ValueError(f"Unknown dataset: {dataset_name}, choices: {list(DATASET_META.keys())}")
    return DATASET_META[dataset_name]


def get_transforms(dataset_name: str, train: bool) -> transforms.Compose:
    """Return transform based on dataset and train/test mode."""
    _, size, in_channels, _ = get_dataset_meta(dataset_name)
    if dataset_name in ("CIFAR10", "CIFAR100"):
        if train:
            return transforms.Compose([
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if dataset_name in ("MNIST", "FashionMNIST"):
        if train:
            return transforms.Compose([
                transforms.RandomCrop(size, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    raise ValueError(f"Transform not implemented: {dataset_name}")


def _load_dataset(name: str, root: str, transform_train, transform_test):
    """Load dataset; retry once on download failure with a clear hint."""
    def _cifar10():
        return (
            torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train),
            torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test),
        )
    def _cifar100():
        return (
            torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train),
            torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test),
        )
    def _mnist():
        return (
            torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train),
            torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test),
        )
    def _fashion():
        return (
            torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train),
            torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test),
        )
    loaders = {
        "CIFAR10": _cifar10,
        "CIFAR100": _cifar100,
        "MNIST": _mnist,
        "FashionMNIST": _fashion,
    }
    if name not in loaders:
        raise ValueError(f"Dataset not implemented: {name}")
    import os
    import time
    abs_root = os.path.abspath(root)
    last_err = None
    for attempt in range(2):
        try:
            return loaders[name]()
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(2)
                continue
            if "not found or corrupted" in str(last_err) or "download" in str(last_err).lower():
                hint = (
                    "CIFAR-100 can be manually downloaded and extracted to the above directory from: https://www.cs.toronto.edu/~kriz/cifar.html"
                    if name == "CIFAR100" else "Please check your network or manually download the dataset to the above directory"
                )
                raise RuntimeError(
                    f"{name} dataset not found or download failed (directory: {abs_root}). {hint}\nOriginal error: {last_err}"
                ) from last_err
            raise RuntimeError(f"{name} load failed (root={abs_root}): {last_err}") from last_err
    raise RuntimeError(f"{name} load failed (root={abs_root}): {last_err}") from last_err


def build_dataloaders(cfg: ExperimentConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/test DataLoader from cfg.dataset and cfg.data_root."""
    name = getattr(cfg, "dataset", "CIFAR10") or "CIFAR10"
    root = cfg.data_root if cfg.data_root else get_dataset_meta(name)[3]
    transform_train = get_transforms(name, train=True)
    transform_test = get_transforms(name, train=False)

    trainset, testset = _load_dataset(name, root, transform_train, transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return trainloader, testloader
