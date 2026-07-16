# -*- coding: utf-8 -*-
"""
Batch experiment config: run_batch_all.py reads from here by default.
Modify the settings below, then run: python run_batch_all.py
"""

# ========== Available models / Available datasets (add/remove here) ==========
# Probabilistic networks use the "Prob" prefix; non-probabilistic use original names for CIFAR10/CIFAR100 baselines
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

# ========== Activation functions (for Prob* model comparison experiments) ==========
ACTIVATION_CHOICES = ["relu", "sigmoid", "tanh", "tanh_sigmoid", "silu", "gelu", "leaky_relu"]
# None = use only default tanh_sigmoid; otherwise specify a list of activations to compare, combined with model x dataset
# ACTIVATIONS = None
# Example (compare multiple activations):
ACTIVATIONS = ["relu", "sigmoid", "tanh", "tanh_sigmoid", "silu", "gelu", "leaky_relu"]

# ========== Parallelism ==========
# Number of experiments to run concurrently (recommended <= number of GPUs)
MAX_WORKERS = 7

# ========== Model list ==========
# None = use all MODEL_CHOICES above; otherwise specify the list of model names to run
# MODELS = None
# Example (run a subset):
# MODELS = ["ProbResNet18", "ProbVGG16", "ProbGoogLeNet"]   # probabilistic
# MODELS = ["VGG16", "DenseNet121"]   # non-probabilistic baselines (CIFAR10/CIFAR100 comparison)
# MODELS = ["ProbVGG16"]
# MODELS = ["VGG16"]
# MODELS = ["GoogLeNet", "ResNet18", "ResNet34","VGG16"]
# MODELS = ["ProbVGG19","VGG19"]
MODELS = ["ProbGoogLeNet"]

# ========== Dataset list ==========
# None = use all DATASET_CHOICES above; otherwise specify the list of dataset names to run
# DATASETS = None
# Example (run a subset):
DATASETS = ["CIFAR10","CIFAR100"]

# ========== GPU ==========
# GPU IDs to use, assigned to experiments in round-robin order
GPU_IDS = [0, 1, 2, 3, 4, 5, 6]

# ========== Optional: training params and output directory ==========
EPOCHS = 100
BATCH_SIZE = 128
OUTPUT_ROOT = "outputs"
# LR = 1e-3   # Uses run_train default if unset
