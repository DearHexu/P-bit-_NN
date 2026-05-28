# -*- coding: utf-8 -*-
"""
批量跑全量实验：所有模型 × 所有数据集。
默认从 config/batch_run_config.py 读取：并行数量、模型列表、数据集列表（改那个文件即可）。
"""

import os
import sys
import subprocess
import multiprocessing
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.default import MODEL_CHOICES, DATASET_CHOICES
from config.batch_run_config import ACTIVATION_CHOICES

# 支持激活函数配置的模型（概率化 Prob*、二值权重 Bin*，做 模型×数据集×激活 对比）
MODELS_WITH_ACTIVATION = [
    "ProbGoogLeNet", "BinGoogLeNet",
    "ProbResNet18", "ProbResNet34", "ProbResNet50", "ProbResNet101", "ProbResNet152",
    "BinResNet18", "BinResNet34", "BinResNet50", "BinResNet101", "BinResNet152",
    "ProbVGG11", "ProbVGG13", "ProbVGG16", "ProbVGG19", "ProbAlexNet",
    "ProbLeNet5", "ProbLeNet3",
    "ProbDenseNet121", "ProbDenseNet169", "ProbDenseNet201", "ProbDenseNet161",
]


def load_batch_config() -> dict:
    """从 config/batch_run_config.py 读取配置；缺少的项用默认值。"""
    try:
        from config import batch_run_config as c
        return {
            "max_workers": getattr(c, "MAX_WORKERS", 2),
            "models": getattr(c, "MODELS", None),
            "datasets": getattr(c, "DATASETS", None),
            "activations": getattr(c, "ACTIVATIONS", None),
            "gpu_ids": getattr(c, "GPU_IDS", [0]),
            "epochs": getattr(c, "EPOCHS", 100),
            "batch_size": getattr(c, "BATCH_SIZE", 128),
            "lr": getattr(c, "LR", None),
            "output_root": getattr(c, "OUTPUT_ROOT", "outputs"),
        }
    except Exception:
        return {
            "max_workers": 2,
            "models": None,
            "datasets": None,
            "activations": None,
            "gpu_ids": [0],
            "epochs": 100,
            "batch_size": 128,
            "lr": None,
            "output_root": "outputs",
        }


def run_one_experiment(exp: Dict[str, Any], output_root: str = "outputs") -> int:
    """子进程运行单次实验；通过 CUDA_VISIBLE_DEVICES 绑定物理 GPU，子进程内只看到 cuda:0。"""
    gpu_id = exp.get("gpu_id", 0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PBT_PHYSICAL_GPU_ID"] = str(gpu_id)  # 供 run_train 打日志用
    cmd = [
        sys.executable,
        os.path.join(ROOT, "run_train.py"),
        "--exp_name", str(exp.get("exp_name", "exp")),
        "--model_name", str(exp.get("model_name", "ProbGoogLeNet")),
        "--dataset", str(exp.get("dataset", "CIFAR10")),
        "--output_root", output_root,
        "--gpu_id", "0",
    ]
    if "epochs" in exp:
        cmd += ["--epochs", str(exp["epochs"])]
    if "batch_size" in exp:
        cmd += ["--batch_size", str(exp["batch_size"])]
    if "lr" in exp:
        cmd += ["--lr", str(exp["lr"])]
    if "activation" in exp:
        cmd += ["--activation", str(exp["activation"])]
    proc = subprocess.run(cmd, env=env, cwd=ROOT)
    return proc.returncode


def build_all_experiments(
    models: List[str],
    datasets: List[str],
    gpu_ids: List[int],
    activations: List[str] = None,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = None,
) -> List[Dict[str, Any]]:
    """
    生成实验列表。若 activations 非空，则对 MODELS_WITH_ACTIVATION 内的模型展开 模型×数据集×激活；
    其余模型仅 模型×数据集，使用默认激活（Prob* 用 tanh_sigmoid，其它不传）。
    """
    experiments = []
    # 先展开 (model, dataset, activation)
    triples = []
    for model_name in models:
        for dataset in datasets:
            if activations and model_name in MODELS_WITH_ACTIVATION:
                for act in activations:
                    triples.append((model_name, dataset, act))
            else:
                triples.append((model_name, dataset, None))
    for i, item in enumerate(triples):
        model_name, dataset, act = item
        gpu_id = gpu_ids[i % len(gpu_ids)]
        # 支持激活的模型：目录名始终带激活，便于从文件夹名看出用的哪种激活
        if act:
            exp_name = f"{model_name}_{dataset}_{act}"
            exp = {"exp_name": exp_name, "model_name": model_name, "dataset": dataset, "activation": act, "gpu_id": gpu_id, "epochs": epochs, "batch_size": batch_size}
        else:
            default_act = "tanh_sigmoid" if model_name in MODELS_WITH_ACTIVATION else None
            exp_name = f"{model_name}_{dataset}_{default_act}" if default_act else f"{model_name}_{dataset}"
            exp = {"exp_name": exp_name, "model_name": model_name, "dataset": dataset, "gpu_id": gpu_id, "epochs": epochs, "batch_size": batch_size}
            if default_act:
                exp["activation"] = default_act
        if lr is not None:
            exp["lr"] = lr
        experiments.append(exp)
    return experiments


def main():
    import argparse
    cfg = load_batch_config()
    parser = argparse.ArgumentParser(
        description="批量运行 模型×数据集 实验。默认从 config/batch_run_config.py 读配置。"
    )
    parser.add_argument("--models", type=str, default=None,
                        help="逗号分隔的模型名，不传则用配置文件中的 MODELS")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集，不传则用配置文件中的 DATASETS")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="逗号分隔的 GPU ID，不传则用配置文件中的 GPU_IDS")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=None,
                        help="并行进程数，不传则用配置文件中的 MAX_WORKERS")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # 命令行覆盖配置文件
    models = [s.strip() for s in args.models.split(",")] if args.models else (cfg["models"] or MODEL_CHOICES)
    datasets = [s.strip() for s in args.datasets.split(",")] if args.datasets else (cfg["datasets"] or DATASET_CHOICES)
    gpu_ids = [int(s.strip()) for s in args.gpu_ids.split(",")] if args.gpu_ids else cfg["gpu_ids"]
    output_root = args.output_root if args.output_root is not None else cfg["output_root"]
    max_workers = args.max_workers if args.max_workers is not None else cfg["max_workers"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    activations = cfg.get("activations")  # None 或列表，用于对比多种激活

    # 过滤非法模型/数据集/激活
    models = [m for m in models if m in MODEL_CHOICES]
    datasets = [d for d in datasets if d in DATASET_CHOICES]
    if activations:
        activations = [a for a in activations if a in ACTIVATION_CHOICES]
    if not models or not datasets:
        print("模型或数据集为空或无效。MODEL_CHOICES:", MODEL_CHOICES, "DATASET_CHOICES:", DATASET_CHOICES)
        return

    experiments = build_all_experiments(
        models, datasets, gpu_ids, activations=activations,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
    n_workers = max_workers or len(gpu_ids)
    n_workers = min(n_workers, len(experiments))
    print(f"共 {len(experiments)} 个实验 (models={len(models)}, datasets={len(datasets)})，并行数 {n_workers}，GPUs: {gpu_ids}")
    for ex in experiments:
        print(f"  -> {ex['exp_name']} -> 物理 GPU {ex['gpu_id']}")

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = []
        for exp in experiments:
            r = pool.apply_async(run_one_experiment, (exp, output_root))
            results.append((exp["exp_name"], r))
        for name, r in results:
            try:
                code = r.get()
                print(f"[{name}] 退出码: {code}")
            except Exception as e:
                print(f"[{name}] 异常: {e}")

    print("配置说明: 修改 config/batch_run_config.py 可改 并行数量 / 模型列表 / 数据集列表")


if __name__ == "__main__":
    main()
