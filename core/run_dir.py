# -*- coding: utf-8 -*-
"""
Experiment run directory management.
Each training run creates a timestamped subdirectory under output_root, holding checkpoints, logs, curves, and metrics.
"""

import os
from datetime import datetime
from typing import Tuple


def create_run_dir(output_root: str, exp_name: str) -> Tuple[str, str, str]:
    """
    Create the output directory structure for this experiment.

    Returns:
        run_dir: root directory for this experiment, e.g. outputs/20250305_143022_ProbGoogLeNet_CIFAR10
        ckpt_dir: checkpoint subdirectory
        log_dir: directory for TensorBoard and train.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace spaces or path characters in experiment name with underscores
    safe_name = exp_name.replace(" ", "_").replace("/", "_")
    run_dir = os.path.join(output_root, f"{timestamp}_{safe_name}")
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return run_dir, ckpt_dir, log_dir
