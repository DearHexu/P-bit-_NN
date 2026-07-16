# -*- coding: utf-8 -*-
"""
Training and testing loop: decoupled from config and logging, shared by single-machine and multi-GPU parallel scripts.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Any

# Avoid circular import: only used for type annotations
from config.default import ExperimentConfig


def time_sync() -> float:
    """Current time after GPU sync (seconds), used for timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def build_model(cfg: ExperimentConfig, device: torch.device) -> nn.Module:
    """Build model from config and move to device; LeNet auto-sets in_channels from dataset."""
    dataset = getattr(cfg, "dataset", "CIFAR10") or "CIFAR10"
    num_classes = cfg.num_classes
    try:
        from core.datasets import get_dataset_meta
        in_channels = get_dataset_meta(dataset)[2]
    except Exception:
        in_channels = 3

    activation = getattr(cfg, "activation", "tanh_sigmoid") or "tanh_sigmoid"
    # ---------- Fully binary-weight models (Bin prefix) ----------
    if cfg.model_name == "BinGoogLeNet":
        from backbones.BinGoogLeNet import BinGoogLeNet
        model = BinGoogLeNet(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "BinResNet18":
        from backbones.BinResNet import BinResNet18
        model = BinResNet18(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "BinResNet34":
        from backbones.BinResNet import BinResNet34
        model = BinResNet34(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "BinResNet50":
        from backbones.BinResNet import BinResNet50
        model = BinResNet50(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "BinResNet101":
        from backbones.BinResNet import BinResNet101
        model = BinResNet101(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "BinResNet152":
        from backbones.BinResNet import BinResNet152
        model = BinResNet152(num_classes=num_classes, activation=activation)
    # ---------- Probabilistic models (Prob prefix) ----------
    elif cfg.model_name == "ProbGoogLeNet":
        from backbones.ProbGoogLeNet import ProbGoogLeNet
        model = ProbGoogLeNet(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbResNet18":
        from backbones.ProbResNet import ProbResNet18
        model = ProbResNet18(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbResNet34":
        from backbones.ProbResNet import ProbResNet34
        model = ProbResNet34(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbResNet50":
        from backbones.ProbResNet import ProbResNet50
        model = ProbResNet50(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbResNet101":
        from backbones.ProbResNet import ProbResNet101
        model = ProbResNet101(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbResNet152":
        from backbones.ProbResNet import ProbResNet152
        model = ProbResNet152(num_classes=num_classes, activation=activation)
    elif cfg.model_name in ("ProbVGG11", "ProbVGG13", "ProbVGG16", "ProbVGG19"):
        from backbones.VGG import ProbVGG
        vgg_name = cfg.model_name[4:]  # Strip "Prob" prefix
        model = ProbVGG(vgg_name, num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbAlexNet":
        from backbones.AlexNet import ProbAlexNet
        model = ProbAlexNet(num_classes=num_classes, activation=activation)
    elif cfg.model_name == "ProbLeNet5":
        from backbones.SimpleCNN import ProbLeNet5
        model = ProbLeNet5(num_class=num_classes, in_channels=in_channels, activation=activation)
    elif cfg.model_name == "ProbLeNet3":
        from backbones.SimpleCNN import ProbLeNet3
        model = ProbLeNet3(num_class=num_classes, in_channels=in_channels, activation=activation)
    elif cfg.model_name in ("ProbDenseNet121", "ProbDenseNet169", "ProbDenseNet201", "ProbDenseNet161"):
        from backbones.DenseNet import ProbDenseNet121, ProbDenseNet169, ProbDenseNet201, ProbDenseNet161
        model = {
            "ProbDenseNet121": ProbDenseNet121, "ProbDenseNet169": ProbDenseNet169,
            "ProbDenseNet201": ProbDenseNet201, "ProbDenseNet161": ProbDenseNet161,
        }[cfg.model_name](num_classes=num_classes, activation=activation)
    # ---------- Non-probabilistic models (original network names, for CIFAR10/CIFAR100 comparison baselines) ----------
    elif cfg.model_name == "GoogLeNet":
        from backbones.ProbGoogLeNet import GoogLeNet
        model = GoogLeNet(num_classes=num_classes)
    elif cfg.model_name in ("ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"):
        from backbones.ProbResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        model = {
            "ResNet18": ResNet18, "ResNet34": ResNet34, "ResNet50": ResNet50,
            "ResNet101": ResNet101, "ResNet152": ResNet152,
        }[cfg.model_name](num_classes=num_classes)
    elif cfg.model_name in ("VGG11", "VGG13", "VGG16", "VGG19"):
        from backbones.VGG import VGG
        model = VGG(cfg.model_name, num_classes=num_classes)
    elif cfg.model_name == "AlexNet":
        from backbones.AlexNet import AlexNet
        model = AlexNet(num_classes=num_classes)
    elif cfg.model_name == "LeNet5":
        from backbones.SimpleCNN import LeNet5
        model = LeNet5(num_class=num_classes, in_channels=in_channels)
    elif cfg.model_name == "LeNet3":
        from backbones.SimpleCNN import LeNet3
        model = LeNet3(num_class=num_classes, in_channels=in_channels)
    elif cfg.model_name in ("DenseNet121", "DenseNet169", "DenseNet201", "DenseNet161"):
        from backbones.DenseNet import DenseNet121, DenseNet169, DenseNet201, DenseNet161
        model = {
            "DenseNet121": DenseNet121, "DenseNet169": DenseNet169,
            "DenseNet201": DenseNet201, "DenseNet161": DenseNet161,
        }[cfg.model_name](num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")
    model = model.to(device)
    if cfg.use_data_parallel and device.type == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model


def build_criterion(cfg: ExperimentConfig) -> nn.Module:
    """Build loss function from config."""
    if cfg.criterion == "CrossEntropy":
        return nn.CrossEntropyLoss()
    if cfg.criterion == "BCEWithLogits":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def build_optimizer_scheduler(model: nn.Module, cfg: ExperimentConfig) -> Tuple[optim.Optimizer, Any]:
    """Build optimizer and LR scheduler. Bin* models: no weight decay, higher learning rate."""
    is_bin = getattr(cfg, "model_name", "").startswith("Bin")
    weight_decay = 0.0 if is_bin else cfg.weight_decay
    lr = cfg.lr * 2.0 if is_bin else cfg.lr
    if cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif cfg.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4 if not is_bin else 0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max)
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_classes: int,
    log_interval: int = 100,
    tb_logger: Optional[Any] = None,
    global_step: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Train one epoch.
    Returns:
        (average train_loss, train_acc)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    step = (global_step or {}).get("step", 0)
    use_one_hot = isinstance(criterion, nn.BCEWithLogitsLoss)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if use_one_hot:
            labels_one_hot = torch.zeros(targets.size(0), num_classes, device=device)
            labels_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            loss = criterion(outputs, labels_one_hot)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = correct / total

        if tb_logger is not None and (batch_idx + 1) % log_interval == 0:
            tb_logger.log_batch(step + batch_idx + 1, loss.item(), train_acc, prefix="train")
        if (batch_idx + 1) % log_interval == 0 and hasattr(train_one_epoch, "_log"):
            train_one_epoch._log(epoch + 1, batch_idx + 1, loss.item(), train_acc)
    if global_step is not None:
        global_step["step"] = step + len(trainloader)
    avg_loss = train_loss / len(trainloader)
    return avg_loss, correct / total


def test_one_epoch(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_classes: int,
) -> Tuple[float, float]:
    """Validate/test one epoch. Returns: (average test_loss, test_acc)."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    use_one_hot = isinstance(criterion, nn.BCEWithLogitsLoss)
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if use_one_hot:
                labels_one_hot = torch.zeros(targets.size(0), num_classes, device=device)
                labels_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                loss = criterion(outputs, labels_one_hot)
            else:
                loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / len(testloader), correct / total


def save_checkpoint(
    model: nn.Module,
    acc: float,
    epoch: int,
    path: str,
):
    """Save current best checkpoint."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "net": model.state_dict(),
        "acc": acc,
        "epoch": epoch,
    }
    torch.save(state, path)
