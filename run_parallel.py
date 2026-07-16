# -*- coding: utf-8 -*-
"""
Multi-config, multi-GPU parallel training entry point.
Distributes multiple experiment configs across different GPUs to run simultaneously,
for comparing different models/hyperparams (e.g. tanh*sigmoid prob net vs others).
Configs specified inline or via YAML/JSON; each experiment runs in a separate process,
bound to a GPU via CUDA_VISIBLE_DEVICES.
"""

import os
import sys
import subprocess
import multiprocessing
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def get_default_experiments() -> List[Dict[str, Any]]:
    """
    Default experiment list: used to validate tanh*sigmoid composite activation
    performance on probabilistic computing networks.
    Each entry specifies: exp_name, model_name, gpu_id, and optionally lr, epochs, batch_size, etc.
    """
    return [
        {"exp_name": "ProbGoogLeNet_CIFAR10", "model_name": "ProbGoogLeNet", "gpu_id": 0},
        {"exp_name": "ProbResNet18_CIFAR10", "model_name": "ProbResNet18", "gpu_id": 1},
        # If only a single GPU, use the same gpu_id (runs sequentially); or keep only one entry.
    ]


def run_one_experiment(exp: Dict[str, Any], output_root: str = "outputs") -> int:
    """
    Run a single experiment in a subprocess.
    Sets CUDA_VISIBLE_DEVICES to exp["gpu_id"]; subprocess uses --gpu_id 0.
    """
    gpu_id = exp.get("gpu_id", 0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "run_train.py"),
        "--exp_name", str(exp.get("exp_name", "exp")),
        "--model_name", str(exp.get("model_name", "ProbGoogLeNet")),
        "--output_root", output_root,
        "--gpu_id", "0",  # Subprocess only sees one GPU — the one assigned above
    ]
    if "dataset" in exp:
        cmd += ["--dataset", str(exp["dataset"])]
    if "epochs" in exp:
        cmd += ["--epochs", str(exp["epochs"])]
    if "batch_size" in exp:
        cmd += ["--batch_size", str(exp["batch_size"])]
    if "lr" in exp:
        cmd += ["--lr", str(exp["lr"])]
    proc = subprocess.run(cmd, env=env, cwd=ROOT)
    return proc.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-config multi-GPU parallel training")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Optional: path to JSON file, one experiment dict per line (with exp_name, model_name, gpu_id, etc.)")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Max number of parallel processes, defaults to number of experiments")
    args = parser.parse_args()

    if args.experiments and os.path.isfile(args.experiments):
        import json
        experiments = []
        with open(args.experiments, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                experiments.append(json.loads(line))
    else:
        experiments = get_default_experiments()

    if not experiments:
        print("No experiment configs, exiting")
        return
    n = min(args.max_workers or len(experiments), len(experiments))
    print(f"Total {len(experiments)} experiments, parallelism {n}, output root: {args.output_root}")

    with multiprocessing.Pool(processes=n) as pool:
        results = []
        for exp in experiments:
            r = pool.apply_async(run_one_experiment, (exp, args.output_root))
            results.append((exp.get("exp_name", "?"), r))
        for name, r in results:
            try:
                code = r.get()
                print(f"[{name}] exit code: {code}")
            except Exception as e:
                print(f"[{name}] exception: {e}")


if __name__ == "__main__":
    main()
