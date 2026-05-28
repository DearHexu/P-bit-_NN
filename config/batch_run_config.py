# -*- coding: utf-8 -*-
"""
批量实验配置：run_batch_all.py 默认从这里读取。
改下面几项后直接运行:  python run_batch_all.py
"""

# ========== 可选模型 / 可选数据集（可在此增删） ==========
# 概率化网络带 Prob 前缀；非概率化用原网络名，用于 CIFAR10/CIFAR100 对比基线
MODEL_CHOICES = [
    "ProbGoogLeNet", "GoogLeNet", "BinGoogLeNet",
    "ProbResNet18", "ProbResNet34", "ProbResNet50", "ProbResNet101", "ProbResNet152",
    "BinResNet18", "BinResNet34", "BinResNet50", "BinResNet101", "BinResNet152",
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
    "ProbVGG11", "ProbVGG13", "ProbVGG16", "ProbVGG19",
    "VGG11", "VGG13", "VGG16", "VGG19",
    "ProbAlexNet", "AlexNet",
    "ProbLeNet5", "LeNet5", "ProbLeNet3", "LeNet3",
    "ProbDenseNet121", "ProbDenseNet169", "ProbDenseNet201", "ProbDenseNet161",
    "DenseNet121", "DenseNet169", "DenseNet201", "DenseNet161",
]
DATASET_CHOICES = ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"]

# ========== 激活函数（用于 Prob* 等模型的对比实验） ==========
ACTIVATION_CHOICES = ["relu", "sigmoid", "tanh", "tanh_sigmoid", "silu", "gelu", "leaky_relu"]
# None = 仅用默认 tanh_sigmoid；否则写要对比的激活列表，与 模型×数据集 组合成实验
# ACTIVATIONS = None
# 示例（对比多种激活）:
ACTIVATIONS = ["relu", "sigmoid", "tanh", "tanh_sigmoid", "silu", "gelu", "leaky_relu"]

# ========== 并行数量 ==========
# 同时跑几个实验（建议 ≤ 显卡数量）
MAX_WORKERS = 7

# ========== 模型列表 ==========
# None = 使用上面 MODEL_CHOICES 全部；否则写要跑的模型名列表
# MODELS = None
# 示例（只跑部分）:
# MODELS = ["ProbResNet18", "ProbVGG16", "ProbGoogLeNet"]   # 概率化
# MODELS = ["VGG16", "DenseNet121"]   # 非概率化基线（CIFAR10/CIFAR100 对比）
# MODELS = ["ProbVGG16"]
# MODELS = ["VGG16"]
# MODELS = ["GoogLeNet", "ResNet18", "ResNet34","VGG16"]
# MODELS = ["ProbVGG19","VGG19"]
MODELS = ["ProbGoogLeNet"]

# ========== 数据集列表 ==========
# None = 使用上面 DATASET_CHOICES 全部；否则写要跑的数据集名列表
# DATASETS = None
# 示例（只跑部分）:
DATASETS = ["CIFAR10","CIFAR100"]

# ========== GPU ==========
# 使用的 GPU 编号，轮询分配给各实验
GPU_IDS = [0, 1, 2, 3, 4, 5, 6]

# ========== 可选：训练参数与输出目录 ==========
EPOCHS = 100
BATCH_SIZE = 128
OUTPUT_ROOT = "outputs"
# LR = 1e-3   # 不写则用 run_train 默认
