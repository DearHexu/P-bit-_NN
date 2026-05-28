# -*- coding: utf-8 -*-
"""
实验运行目录管理。
每次训练在 output_root 下创建带时间戳与实验名的子目录，统一存放 checkpoint、日志、曲线与指标。
"""

import os
from datetime import datetime
from typing import Tuple


def create_run_dir(output_root: str, exp_name: str) -> Tuple[str, str, str]:
    """
    创建本次实验的输出目录结构。

    Returns:
        run_dir: 本次实验根目录，如 outputs/20250305_143022_ProbGoogLeNet_CIFAR10
        ckpt_dir: checkpoint 子目录
        log_dir: TensorBoard 与 train.log 所在目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 实验名中若有空格或路径字符，替换为下划线
    safe_name = exp_name.replace(" ", "_").replace("/", "_")
    run_dir = os.path.join(output_root, f"{timestamp}_{safe_name}")
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return run_dir, ckpt_dir, log_dir
