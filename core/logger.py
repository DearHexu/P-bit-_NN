# -*- coding: utf-8 -*-
"""
Training logger: TensorBoard real-time curves + file logging.
View loss/accuracy changes in the browser in real time, like TensorFlow.
"""

import os
import logging
from typing import Optional

# TensorBoard (built into PyTorch, same format as TensorFlow)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except Exception:
    HAS_TB = False
    SummaryWriter = None


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logger: console + optional file.
    The returned logger is used to print training info (complementary to TensorBoard).
    """
    log = logging.getLogger("pbitnet")
    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def get_logger(name: str = "pbitnet") -> logging.Logger:
    """Get the configured logger."""
    return logging.getLogger(name)


class TensorBoardLogger:
    """
    TensorBoard logger wrapper: records scalars such as loss, accuracy for real-time curve viewing.
    Usage: call add_scalar / log_epoch in the train/valid loop.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled and HAS_TB
        self._writer = None
        if self.enabled:
            self._writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
            self._writer.flush()

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  test_loss: Optional[float] = None, test_acc: Optional[float] = None):
        """Record train/test loss and accuracy per epoch."""
        self.add_scalar("loss/train", train_loss, epoch)
        self.add_scalar("accuracy/train", train_acc, epoch)
        if test_loss is not None:
            self.add_scalar("loss/test", test_loss, epoch)
        if test_acc is not None:
            self.add_scalar("accuracy/test", test_acc, epoch)

    def log_batch(self, global_step: int, loss: float, acc: float, prefix: str = "train"):
        """Record per-batch loss/acc by step (optional, for finer-grained curves)."""
        self.add_scalar(f"batch/{prefix}_loss", loss, global_step)
        self.add_scalar(f"batch/{prefix}_acc", acc, global_step)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
