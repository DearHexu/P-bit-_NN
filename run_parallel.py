# -*- coding: utf-8 -*-
"""
多配置、多 GPU 并行训练入口。
将多组实验配置分配到不同显卡上同时运行，用于对比不同模型/超参（如 tanh*sigmoid 概率网络 vs 其他）。
配置在脚本内或通过 YAML/JSON 指定；每个实验在独立进程中运行，通过 CUDA_VISIBLE_DEVICES 绑定 GPU。
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
    默认实验列表：用于验证 tanh*sigmoid 复合激活在概率计算网络上的性能。
    每项指定：exp_name, model_name, gpu_id, 以及可选的 lr, epochs, batch_size 等。
    """
    return [
        {"exp_name": "ProbGoogLeNet_CIFAR10", "model_name": "ProbGoogLeNet", "gpu_id": 0},
        {"exp_name": "ProbResNet18_CIFAR10", "model_name": "ProbResNet18", "gpu_id": 1},
        # 若只有单卡，可改为相同 gpu_id，将顺序执行；或只保留一条。
    ]


def run_one_experiment(exp: Dict[str, Any], output_root: str = "outputs") -> int:
    """
    在子进程中运行单次实验。
    设置 CUDA_VISIBLE_DEVICES 为 exp["gpu_id"]，子进程内使用 --gpu_id 0。
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
        "--gpu_id", "0",  # 子进程内只看到一块卡，即上面指定的那块
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
    parser = argparse.ArgumentParser(description="多配置多 GPU 并行训练")
    parser.add_argument("--experiments", type=str, default=None,
                        help="可选：JSON 文件路径，每行一个实验 dict（含 exp_name, model_name, gpu_id 等）")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="最大并行进程数，默认等于实验数")
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
        print("没有实验配置，退出")
        return
    n = min(args.max_workers or len(experiments), len(experiments))
    print(f"共 {len(experiments)} 个实验，并行数 {n}，输出根目录: {args.output_root}")

    with multiprocessing.Pool(processes=n) as pool:
        results = []
        for exp in experiments:
            r = pool.apply_async(run_one_experiment, (exp, args.output_root))
            results.append((exp.get("exp_name", "?"), r))
        for name, r in results:
            try:
                code = r.get()
                print(f"[{name}] 退出码: {code}")
            except Exception as e:
                print(f"[{name}] 异常: {e}")


if __name__ == "__main__":
    main()
