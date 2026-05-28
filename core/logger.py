# -*- coding: utf-8 -*-
"""
训练日志：TensorBoard 实时曲线 + 文件日志。
便于像 TensorFlow 一样在浏览器中实时查看 loss/accuracy 变化。
"""

import os
import logging
from typing import Optional

# TensorBoard（PyTorch 内置，与 TensorFlow 同格式）
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except Exception:
    HAS_TB = False
    SummaryWriter = None


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    配置根 logger：控制台 + 可选文件。
    返回的 logger 用于打印训练信息（与 TensorBoard 互补）。
    """
    log = logging.getLogger("pbitnet")
    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # 控制台
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
    """获取已配置的 logger。"""
    return logging.getLogger(name)


class TensorBoardLogger:
    """
    TensorBoard 记录器封装：记录 loss、accuracy 等标量，便于实时查看曲线。
    使用方式：在 train/valid 循环中调用 add_scalar / log_epoch。
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
        """按 epoch 记录训练/测试 loss 与准确率。"""
        self.add_scalar("loss/train", train_loss, epoch)
        self.add_scalar("accuracy/train", train_acc, epoch)
        if test_loss is not None:
            self.add_scalar("loss/test", test_loss, epoch)
        if test_acc is not None:
            self.add_scalar("accuracy/test", test_acc, epoch)

    def log_batch(self, global_step: int, loss: float, acc: float, prefix: str = "train"):
        """按 step 记录当前 batch 的 loss/acc（可选，用于更细粒度曲线）。"""
        self.add_scalar(f"batch/{prefix}_loss", loss, global_step)
        self.add_scalar(f"batch/{prefix}_acc", acc, global_step)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
