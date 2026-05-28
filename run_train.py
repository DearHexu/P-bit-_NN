# -*- coding: utf-8 -*-
"""
PbitNet 训练入口（单机单实验）。
- 使用 config 统一参数，支持命令行覆盖。
- 日志：TensorBoard（实时 loss/acc 曲线）+ 文件 train.log。
- 结果统一落在 outputs/<时间戳>_<实验名>/ 下：checkpoint、logs、metrics.csv、accuracy.png。
"""

import os
import sys
import json
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.default import ExperimentConfig, get_config, parse_args
from core.run_dir import create_run_dir
from core.logger import setup_logging, get_logger, TensorBoardLogger
from core.datasets import build_dataloaders as _build_dataloaders, get_dataset_meta
from core.train_loop import (
    time_sync,
    build_model,
    build_criterion,
    build_optimizer_scheduler,
    train_one_epoch,
    test_one_epoch,
    save_checkpoint,
)


def build_dataloaders(cfg: ExperimentConfig):
    """根据 cfg.dataset 构建 DataLoader。"""
    return _build_dataloaders(cfg)


def run(cfg: ExperimentConfig):
    """单次实验完整流程：创建目录、日志、数据、模型、训练、保存结果。"""
    gpu_id = cfg.gpu_id if cfg.gpu_id is not None else 0
    physical_gpu = os.environ.get("PBT_PHYSICAL_GPU_ID")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    run_dir, ckpt_dir, log_dir = create_run_dir(cfg.output_root, cfg.exp_name)
    train_log_file = os.path.join(log_dir, "train.log")
    setup_logging(train_log_file)
    log = get_logger()
    tb = TensorBoardLogger(log_dir, enabled=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "last.pth")

    device_str = f"{device} (物理 GPU {physical_gpu})" if physical_gpu else str(device)
    activation = getattr(cfg, "activation", "tanh_sigmoid") or "tanh_sigmoid"
    log.info("实验名: %s, 模型: %s, 激活: %s, 设备: %s", cfg.exp_name, cfg.model_name, activation, device_str)
    log.info("输出目录: %s", run_dir)
    log.info("TensorBoard 日志: %s  （看曲线: tensorboard --logdir=%s）", log_dir, os.path.abspath(cfg.output_root))

    # 保存本实验配置到 run_config.json，便于从结果目录直接看出用的哪种激活等
    run_config = {
        "model_name": cfg.model_name,
        "dataset": cfg.dataset,
        "activation": activation,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "exp_name": cfg.exp_name,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    trainloader, testloader = build_dataloaders(cfg)
    model = build_model(cfg, device)
    criterion = build_criterion(cfg)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    start_epoch = 0
    best_acc = 0.0
    if cfg.resume and cfg.checkpoint_path and os.path.isfile(cfg.checkpoint_path):
        log.info("从 checkpoint 恢复: %s", cfg.checkpoint_path)
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["net"], strict=True)
        best_acc = ckpt.get("acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1

    total_train_acc, total_test_acc = [], []
    global_step = {"step": 0}

    def _batch_log(epoch, batch_idx, loss, acc):
        log.info("[Epoch-%d Batch-%d] Train Loss=%.4f Acc=%.4f", epoch, batch_idx, loss, acc)
    import core.train_loop as tl
    tl.train_one_epoch._log = _batch_log

    t0 = time_sync()
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch, cfg.num_classes,
            log_interval=100, tb_logger=tb, global_step=global_step,
        )
        test_loss, test_acc = test_one_epoch(
            model, testloader, criterion, device, epoch, cfg.num_classes,
        )
        scheduler.step()
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        tb.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)

        acc_pct = 100.0 * test_acc
        log.info("[Epoch-%d] Train Loss=%.4f Acc=%.4f | Test Loss=%.4f Acc=%.4f (%.2f%%)",
                 epoch + 1, train_loss, train_acc, test_loss, test_acc, acc_pct)

        if acc_pct > best_acc:
            best_acc = acc_pct
            save_checkpoint(model, best_acc, epoch, best_ckpt_path)
            log.info("保存最佳 checkpoint -> %s", best_ckpt_path)
    tb.close()

    save_checkpoint(model, best_acc, cfg.epochs - 1, last_ckpt_path)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    df = pd.DataFrame({
        "epoch": range(len(total_train_acc)),
        "train_acc": total_train_acc,
        "test_acc": total_test_acc,
    })
    df.to_csv(metrics_path, index=False)
    # 在 CSV 同目录保存一份带 run 信息的摘要，便于一眼看出激活等
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"model_name={cfg.model_name}\ndataset={cfg.dataset}\nactivation={activation}\nepochs={cfg.epochs}\nbest_test_acc={best_acc:.2f}%\n")
    log.info("指标已保存: %s", metrics_path)
    fig, ax = plt.subplots()
    ax.plot(range(len(total_train_acc)), total_train_acc, label="Train Accuracy")
    ax.plot(range(len(total_test_acc)), total_test_acc, label="Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    activation = getattr(cfg, "activation", "tanh_sigmoid") or "tanh_sigmoid"
    ax.set_title(f"{cfg.exp_name} (activation={activation})")
    ax.legend()
    fig.savefig(os.path.join(run_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("曲线图已保存: %s/accuracy.png", run_dir)
    elapsed = (time_sync() - t0) / 3600.0
    log.info("训练结束. Best Test Acc: %.2f%%, 耗时: %.3f h", best_acc, elapsed)
    return run_dir, best_acc


# 仅概率化/二值权重模型（Prob/Bin 前缀）支持激活配置；非概率化模型 exp_name 不带激活
MODELS_WITH_ACTIVATION = [
    "ProbGoogLeNet", "BinGoogLeNet",
    "ProbResNet18", "ProbResNet34", "ProbResNet50", "ProbResNet101", "ProbResNet152",
    "BinResNet18", "BinResNet34", "BinResNet50", "BinResNet101", "BinResNet152",
    "ProbVGG11", "ProbVGG13", "ProbVGG16", "ProbVGG19", "ProbAlexNet",
    "ProbLeNet5", "ProbLeNet3",
    "ProbDenseNet121", "ProbDenseNet169", "ProbDenseNet201", "ProbDenseNet161",
]


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args)
    cfg.num_classes = get_dataset_meta(getattr(cfg, "dataset", "CIFAR10") or "CIFAR10")[0]
    if not getattr(args, "exp_name", None) or args.exp_name == "":
        act = getattr(cfg, "activation", "tanh_sigmoid") or "tanh_sigmoid"
        if cfg.model_name in MODELS_WITH_ACTIVATION:
            cfg.exp_name = f"{cfg.model_name}_{cfg.dataset}_{act}"
        else:
            cfg.exp_name = f"{cfg.model_name}_{cfg.dataset}"
    run(cfg)
