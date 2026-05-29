# PbitNet — Probabilistic Binary Neural Networks in PyTorch

[English](#english) | [中文](#中文)

---

## English

### Description

**PbitNet** is a PyTorch framework for training and evaluating **probabilistic binary neural networks**. It implements a **composite activation function** `tanh(x) * sigmoid(x)` paired with a **stochastic Straight-Through Estimator** (`BinarySTE`) to approximate binary (0/1) activations via probabilistic sampling. The repository provides:

- **Prob\*** model variants that integrate the tanh×sigmoid + BinarySTE pipeline into standard CNN backbones (GoogLeNet, ResNet, VGG, AlexNet, LeNet, DenseNet).
- **Bin\*** model variants with binarized weights for comparison.
- Baseline **classic CNN** implementations for direct benchmarking.
- Batch experiment orchestration for systematic hyperparameter sweeps across models, datasets, and activation functions.

### Key Concept: tanh×sigmoid + BinarySTE

| Component | Role |
|---|---|
| `tanh(x) * sigmoid(x)` | Smooth, bounded activation that saturates near 0 and 1, approximating a probabilistic binary gate. |
| `BinarySTE` | Stochastic binarization at forward pass (`x > threshold → 0/1`) with identity gradient in the backward pass (Straight-Through Estimator). Supports uniform and Gaussian threshold distributions. |

This combination allows the network to learn binary-like representations while maintaining differentiability during backpropagation.

### Activation Functions

Seven activation functions are supported for comparative experiments (configured per Prob\*/Bin\* model):

`relu` · `sigmoid` · `tanh` · **`tanh_sigmoid`** (default) · `silu` · `gelu` · `leaky_relu`

Visualizations of all activation curves can be generated via:
```bash
python activation_image.py
```

### Model Zoo

**Probabilistic (tanh×sigmoid + BinarySTE):**
`ProbGoogLeNet` · `ProbResNet18` / `34` / `50` / `101` / `152` · `ProbVGG11` / `13` / `16` / `19` · `ProbAlexNet` · `ProbLeNet5` / `ProbLeNet3` · `ProbDenseNet121` / `169` / `201` / `161`

**Binary weight (bias-free binarization):**
`BinGoogLeNet` · `BinResNet18` / `34` / `50` / `101` / `152`

**Classic baselines:**
`GoogLeNet` · `ResNet18` / `34` / `50` / `101` / `152` · `VGG11` / `13` / `16` / `19` · `AlexNet` · `LeNet5` / `LeNet3` · `DenseNet121` / `169` / `201` / `161`

### Dataset Information

| Dataset | Classes | Input Size | Default Root |
|---|---|---|---|
| **CIFAR-10** | 10 | 32×32 RGB | `CIFAR10/` |
| **CIFAR-100** | 100 | 32×32 RGB | `CIFAR100/` |
| **MNIST** | 10 | 28×28 grayscale | `data/` |
| **Fashion-MNIST** | 10 | 28×28 grayscale | `fashion/` |

Datasets are downloaded automatically via `torchvision.datasets` on first use. CIFAR-10/100 archives are also bundled in the repository under `CIFAR10/` and `CIFAR100/`.

### Project Structure

```
.
├── config/
│   ├── default.py              # ExperimentConfig dataclass, argument parsing
│   └── batch_run_config.py     # Batch experiment defaults (models, datasets, GPUs)
├── core/
│   ├── train_loop.py           # train_one_epoch, test_one_epoch, checkpoint I/O
│   ├── datasets.py             # DataLoader builders for all supported datasets
│   ├── logger.py               # File + TensorBoard logging
│   └── run_dir.py              # Output directory scaffolding
├── backbones/
│   ├── ProbGoogLeNet.py        # Prob* / Bin* / baseline GoogLeNet implementations
│   ├── ProbResNet.py           # Prob* / Bin* / baseline ResNet implementations
│   ├── VGG.py                  # ProbVGG* / VGG* implementations
│   ├── AlexNet.py              # ProbAlexNet / AlexNet implementations
│   ├── SimpleCNN.py            # ProbLeNet* / LeNet* implementations
│   ├── DenseNet.py             # ProbDenseNet* / DenseNet* implementations
│   ├── BinarySTE.py            # Stochastic Straight-Through Estimator
│   └── BinaryWeight.py         # Binary weight convolution wrapper
├── run_train.py                # Single-experiment entry point
├── run_parallel.py             # Multi-config multi-GPU parallel runner
├── run_batch_all.py            # Full combinatorial batch runner
├── infer.py                    # Weight distribution visualization
├── activation_image.py         # Activation function plotter
├── binary_weight_analysis.ipynb # Jupyter notebook for weight analysis
└── activation_functions.png    # Pre-generated activation curves
```

### Requirements

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.7 | Runtime |
| PyTorch | ≥ 1.10 | Deep learning framework |
| torchvision | ≥ 0.11 | Dataset loading, transforms |
| tensorboard | ≥ 2.0 | Real-time loss/accuracy curves |
| pandas | ≥ 1.0 | Metrics CSV export |
| matplotlib | ≥ 3.0 | Accuracy plots, weight visualizations |
| numpy | ≥ 1.19 | Numerical utilities |

Install dependencies:
```bash
pip install torch torchvision tensorboard pandas matplotlib numpy
```

### Usage

#### 1. Single Training Run

```bash
python run_train.py --model_name ProbGoogLeNet --dataset CIFAR10 --epochs 100 --gpu_id 0
python run_train.py --model_name VGG16 --dataset CIFAR100 --epochs 50
python run_train.py --model_name LeNet5 --dataset MNIST
```

With a specific activation function (Prob\*/Bin\* models only):
```bash
python run_train.py --model_name ProbResNet18 --activation relu
```

Outputs are saved to `outputs/<timestamp>_<experiment_name>/`:
- `run_config.json` — experiment configuration snapshot
- `summary.txt` — human-readable result summary
- `checkpoint/best.pth` — best-accuracy model weights
- `checkpoint/last.pth` — final-epoch model weights
- `logs/train.log` — text log file
- `logs/events.out.tfevents.*` — TensorBoard event files
- `metrics.csv` — per-epoch train/test accuracy
- `accuracy.png` — accuracy over epochs plot

#### 2. Real-time Monitoring with TensorBoard

```bash
tensorboard --logdir=outputs --port=6006
```

Open `http://localhost:6006` and view loss/accuracy curves under the **SCALARS** tab. If dashboards appear empty, use an absolute path:
```bash
tensorboard --logdir=/absolute/path/to/project/outputs --port=6006
```

#### 3. Multi-Config Multi-GPU Parallel Execution

Edit `get_default_experiments()` in `run_parallel.py` or supply a JSON config file:
```bash
python run_parallel.py --experiments config/experiments_example.json --output_root outputs
```

Each experiment in the JSON is dispatched to an available GPU via a process pool.

#### 4. Full Combinatorial Batch Run

Edit **`config/batch_run_config.py`** to set:

| Parameter | Description |
|---|---|
| `MODELS` | List of model names, or `None` to use all |
| `DATASETS` | List of dataset names, or `None` to use all |
| `ACTIVATIONS` | List of activation functions, or `None` for default only |
| `MAX_WORKERS` | Maximum concurrent experiments |
| `GPU_IDS` | GPU indices to distribute across |

Then launch:
```bash
python run_batch_all.py
```

CLI overrides are available: `--models`, `--datasets`, `--gpu_ids`, `--max_workers`, `--epochs`.

#### 5. Weight Distribution Visualization

```bash
python infer.py --checkpoint outputs/<experiment>/checkpoint/best.pth --model ProbResNet18
```

Generates `weight_distribution.png` showing the histogram of convolutional layer weights.

### Methodology

1. **Model construction.** Each Prob\* model wraps standard CNN layers with the tanh×sigmoid activation followed by BinarySTE, producing binary stochastic outputs at each layer.
2. **Training.** Standard supervised classification with cross-entropy loss. The BinarySTE module uses a straight-through gradient estimator so the non-differentiable binarization step does not block backpropagation.
3. **Evaluation.** Test-set accuracy is recorded per epoch; the best checkpoint is retained. Metrics are exported as CSV and accuracy-over-epoch plots for downstream analysis.
4. **Comparison.** The framework runs identical experiments across model variants (Prob\*, Bin\*, and classic baselines) and activation functions, enabling controlled A/B comparisons.

### License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 中文

### 项目简介

**PbitNet** 是一个基于 PyTorch 的概率二值神经网络训练与评测框架。其核心创新是使用 **复合激活函数** `tanh(x) * sigmoid(x)` 配合 **随机直通估计器**（`BinarySTE`），通过概率采样将激活值量化为 0/1 二值输出，在保持可微性的同时实现类二值表示。

本仓库提供：

- **Prob\*** 系列模型：将 tanh×sigmoid + BinarySTE 管线集成到主流 CNN 骨干网络中。
- **Bin\*** 系列模型：使用二值化权重的对比模型。
- **经典 CNN 基线**：标准浮点模型，用于直接对比。
- **批量实验编排**：支持跨模型、数据集和激活函数的系统化超参数扫描。

### 核心机制：tanh×sigmoid + BinarySTE

| 组件 | 作用 |
|---|---|
| `tanh(x) * sigmoid(x)` | 平滑有界激活函数，输出在 0 附近趋向 0、在 1 附近趋向 1，近似概率二值门。 |
| `BinarySTE` | 前向传播时通过随机阈值将输入二值化为 0/1，反向传播时使用恒等梯度（Straight-Through Estimator）。支持均匀分布和高斯分布两种阈值模式。 |

通过该组合，网络可以学习二值化表征，同时反向传播不受不可微操作的影响。

### 激活函数

支持七种激活函数用于对比实验（Prob\*/Bin\* 模型可配置）：

`relu` · `sigmoid` · `tanh` · **`tanh_sigmoid`**（默认）· `silu` · `gelu` · `leaky_relu`

生成激活函数图像：
```bash
python activation_image.py
```

### 模型库

**概率化模型（tanh×sigmoid + BinarySTE）：**
`ProbGoogLeNet` · `ProbResNet18` / `34` / `50` / `101` / `152` · `ProbVGG11` / `13` / `16` / `19` · `ProbAlexNet` · `ProbLeNet5` / `ProbLeNet3` · `ProbDenseNet121` / `169` / `201` / `161`

**二值权重模型：**
`BinGoogLeNet` · `BinResNet18` / `34` / `50` / `101` / `152`

**经典基线：**
`GoogLeNet` · `ResNet18` / `34` / `50` / `101` / `152` · `VGG11` / `13` / `16` / `19` · `AlexNet` · `LeNet5` / `LeNet3` · `DenseNet121` / `169` / `201` / `161`

### 数据集信息

| 数据集 | 类别数 | 输入尺寸 | 默认路径 |
|---|---|---|---|
| **CIFAR-10** | 10 | 32×32 RGB | `CIFAR10/` |
| **CIFAR-100** | 100 | 32×32 RGB | `CIFAR100/` |
| **MNIST** | 10 | 28×28 灰度 | `data/` |
| **Fashion-MNIST** | 10 | 28×28 灰度 | `fashion/` |

首次使用时通过 `torchvision.datasets` 自动下载。CIFAR-10/100 的压缩包也随仓库附带在 `CIFAR10/` 和 `CIFAR100/` 目录下。

### 项目结构

```
.
├── config/
│   ├── default.py              # ExperimentConfig 数据类、命令行参数解析
│   └── batch_run_config.py     # 批量实验默认配置（模型、数据集、GPU）
├── core/
│   ├── train_loop.py           # 训练/测试循环、checkpoint 存取
│   ├── datasets.py             # 各数据集的 DataLoader 构建
│   ├── logger.py               # 文件日志 + TensorBoard 记录
│   └── run_dir.py              # 输出目录脚手架
├── backbones/
│   ├── ProbGoogLeNet.py        # Prob* / Bin* / 经典 GoogLeNet 实现
│   ├── ProbResNet.py           # Prob* / Bin* / 经典 ResNet 实现
│   ├── VGG.py                  # ProbVGG* / VGG* 实现
│   ├── AlexNet.py              # ProbAlexNet / AlexNet 实现
│   ├── SimpleCNN.py            # ProbLeNet* / LeNet* 实现
│   ├── DenseNet.py             # ProbDenseNet* / DenseNet* 实现
│   ├── BinarySTE.py            # 随机直通估计器
│   └── BinaryWeight.py         # 二值权重卷积包装器
├── run_train.py                # 单实验训练入口
├── run_parallel.py             # 多配置多 GPU 并行跑
├── run_batch_all.py            # 全量组合批量跑
├── infer.py                    # 权重分布可视化
├── activation_image.py         # 激活函数图像绘制
├── binary_weight_analysis.ipynb # 权重分析 Jupyter Notebook
└── activation_functions.png    # 预生成的激活函数对比图
```

### 环境依赖

| 包 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.7 | 运行环境 |
| PyTorch | ≥ 1.10 | 深度学习框架 |
| torchvision | ≥ 0.11 | 数据集加载与预处理 |
| tensorboard | ≥ 2.0 | 实时 loss/accuracy 曲线 |
| pandas | ≥ 1.0 | 指标导出为 CSV |
| matplotlib | ≥ 3.0 | 准确率曲线、权重分布图 |
| numpy | ≥ 1.19 | 数值计算 |

安装命令：
```bash
pip install torch torchvision tensorboard pandas matplotlib numpy
```

### 使用方式

#### 1. 单次训练

```bash
python run_train.py --model_name ProbGoogLeNet --dataset CIFAR10 --epochs 100 --gpu_id 0
python run_train.py --model_name VGG16 --dataset CIFAR100 --epochs 50
python run_train.py --model_name LeNet5 --dataset MNIST
```

指定激活函数（仅 Prob\*/Bin\* 模型）：
```bash
python run_train.py --model_name ProbResNet18 --activation relu
```

结果保存在 `outputs/<时间戳>_<实验名>/` 下：
- `run_config.json` — 实验配置快照
- `summary.txt` — 可读结果摘要
- `checkpoint/best.pth` — 最佳准确率模型权重
- `checkpoint/last.pth` — 最终 epoch 模型权重
- `logs/train.log` — 文本日志
- `logs/events.out.tfevents.*` — TensorBoard 事件文件
- `metrics.csv` — 每个 epoch 的训练/测试准确率
- `accuracy.png` — 准确率随 epoch 变化曲线

#### 2. 实时监控（TensorBoard）

```bash
tensorboard --logdir=outputs --port=6006
```

浏览器打开 `http://localhost:6006`，在 **SCALARS** 选项卡查看 loss/accuracy 曲线。若显示 "No dashboards are active"，使用绝对路径：
```bash
tensorboard --logdir=/项目绝对路径/outputs --port=6006
```

#### 3. 多配置多 GPU 并行

编辑 `run_parallel.py` 中的 `get_default_experiments()` 或使用 JSON 配置文件：
```bash
python run_parallel.py --experiments config/experiments_example.json --output_root outputs
```

各实验通过进程池分配到空闲 GPU 执行。

#### 4. 全量组合批量实验

编辑 **`config/batch_run_config.py`**，调整以下参数：

| 参数 | 说明 |
|---|---|
| `MODELS` | 模型列表，`None` 表示全部 |
| `DATASETS` | 数据集列表，`None` 表示全部 |
| `ACTIVATIONS` | 激活函数列表，`None` 表示仅默认 |
| `MAX_WORKERS` | 最大并行实验数 |
| `GPU_IDS` | 使用的 GPU 编号列表 |

然后运行：
```bash
python run_batch_all.py
```

支持命令行覆盖：`--models`、`--datasets`、`--gpu_ids`、`--max_workers`、`--epochs`。

#### 5. 权重分布可视化

```bash
python infer.py --checkpoint outputs/<实验目录>/checkpoint/best.pth --model ProbResNet18
```

生成 `weight_distribution.png`，展示卷积层权重的直方图分布。

### 实验方法

1. **模型构建。** 每个 Prob\* 模型在标准 CNN 层之后依次应用 tanh×sigmoid 激活函数和 BinarySTE 模块，使每一层输出为概率二值化结果。
2. **训练。** 使用标准交叉熵损失进行监督分类。BinarySTE 通过直通梯度估计器使不可微的二值化步骤不影响反向传播。
3. **评估。** 每个 epoch 记录测试集准确率，保留最佳 checkpoint。指标导出为 CSV 和准确率曲线图，便于后续分析。
4. **对比。** 框架支持在模型变体（Prob\*、Bin\*、经典基线）和激活函数之间运行完全相同的实验配置，确保受控的 A/B 对比。

### 许可证

本作品采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可协议。
