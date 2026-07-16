# -*- coding: utf-8 -*-
"""
Batch run all experiments: all models x all datasets.
Reads from config/batch_run_config.py by default: parallelism, model list, dataset list (just edit that file).
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

# Models that support activation config (Prob* probabilistic, Bin* binary-weight; run model x dataset x activation comparisons)
MODELS_WITH_ACTIVATION = [
    "ProbGoogLeNet", "BinGoogLeNet",
    "ProbResNet18", "ProbResNet34", "ProbResNet50", "ProbResNet101", "ProbResNet152",
    "BinResNet18", "BinResNet34", "BinResNet50", "BinResNet101", "BinResNet152",
    "ProbVGG11", "ProbVGG13", "ProbVGG16", "ProbVGG19", "ProbAlexNet",
    "ProbLeNet5", "ProbLeNet3",
    "ProbDenseNet121", "ProbDenseNet169", "ProbDenseNet201", "ProbDenseNet161",
]


def load_batch_config() -> dict:
    """Read config from config/batch_run_config.py; use defaults for missing items."""
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
    """Run a single experiment in a subprocess; binds a physical GPU via CUDA_VISIBLE_DEVICES, subprocess sees only cuda:0."""
    gpu_id = exp.get("gpu_id", 0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PBT_PHYSICAL_GPU_ID"] = str(gpu_id)  # For run_train logging
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
    Generate experiment list. If activations is non-empty, expand model x dataset x activation for MODELS_WITH_ACTIVATION models;
    other models use only model x dataset, with default activation (Prob* uses tanh_sigmoid, others don't pass it).
    """
    experiments = []
    # First expand (model, dataset, activation)
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
        # Activation-capable models: directory name always includes activation, for easy identification from folder name
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
        description="Batch run model x dataset experiments. Reads config from config/batch_run_config.py by default."
    )
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names; uses config file MODELS if not provided")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated datasets; uses config file DATASETS if not provided")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs; uses config file GPU_IDS if not provided")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of parallel processes; uses config file MAX_WORKERS if not provided")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # CLI overrides config file
    models = [s.strip() for s in args.models.split(",")] if args.models else (cfg["models"] or MODEL_CHOICES)
    datasets = [s.strip() for s in args.datasets.split(",")] if args.datasets else (cfg["datasets"] or DATASET_CHOICES)
    gpu_ids = [int(s.strip()) for s in args.gpu_ids.split(",")] if args.gpu_ids else cfg["gpu_ids"]
    output_root = args.output_root if args.output_root is not None else cfg["output_root"]
    max_workers = args.max_workers if args.max_workers is not None else cfg["max_workers"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    activations = cfg.get("activations")  # None or list, for comparing multiple activations

    # Filter invalid models/datasets/activations
    models = [m for m in models if m in MODEL_CHOICES]
    datasets = [d for d in datasets if d in DATASET_CHOICES]
    if activations:
        activations = [a for a in activations if a in ACTIVATION_CHOICES]
    if not models or not datasets:
        print("Models or datasets empty or invalid. MODEL_CHOICES:", MODEL_CHOICES, "DATASET_CHOICES:", DATASET_CHOICES)
        return

    experiments = build_all_experiments(
        models, datasets, gpu_ids, activations=activations,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
    n_workers = max_workers or len(gpu_ids)
    n_workers = min(n_workers, len(experiments))
    print(f"Total {len(experiments)} experiments (models={len(models)}, datasets={len(datasets)}), parallelism {n_workers}, GPUs: {gpu_ids}")
    for ex in experiments:
        print(f"  -> {ex['exp_name']} -> physical GPU {ex['gpu_id']}")

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = []
        for exp in experiments:
            r = pool.apply_async(run_one_experiment, (exp, output_root))
            results.append((exp["exp_name"], r))
        for name, r in results:
            try:
                code = r.get()
                print(f"[{name}] exit code: {code}")
            except Exception as e:
                print(f"[{name}] exception: {e}")

    print("Config note: edit config/batch_run_config.py to change parallelism / model list / dataset list")


if __name__ == "__main__":
    main()
