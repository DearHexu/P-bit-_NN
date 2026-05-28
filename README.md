# Pytorch PbitNet / PyTorch 概率计算网络

[中文](#中文) | [English](#english)

---

## 中文

基于 PyTorch 的概率计算网络（PbitNet），使用 **tanh 与 sigmoid 复合激活函数**（`tanh(x)*sigmoid(x)`）在多数据集上验证性能。

### 激活函数对比（仅 Prob* 模型）

在 **config/batch_run_config.py** 中可配置：
- **ACTIVATION_CHOICES**：可选激活列表（relu, sigmoid, tanh, **tanh_sigmoid**, silu, gelu, leaky_relu）
- **ACTIVATIONS**：`None` = 仅用默认 tanh_sigmoid；设为列表如 `["relu", "tanh_sigmoid", "silu"]` 时，批量跑会对每个 Prob* 模型 × 数据集 × 激活 各跑一次，便于对比

单次训练也可指定：`python run_train.py --model_name ProbResNet18 --activation relu`

### 支持的模型（config.MODEL_CHOICES）

- **Prob 系列**：ProbGoogLeNet，ProbResNet18/34/50/101/152（含 tanh*sigmoid + BinarySTE）
- **经典 CNN**：AlexNet，VGG11/13/16/19，LeNet5/LeNet3，DenseNet121/169/201/161

### 支持的数据集

- CIFAR10、CIFAR100、MNIST、FashionMNIST

### 项目结构

- **config/**：统一参数配置（MODEL_CHOICES、DATASET_CHOICES、ExperimentConfig）
- **core/**：核心逻辑（train_loop、datasets、logger、run_dir）
- **backbones/**：网络结构（Prob*、AlexNet、VGG、SimpleCNN、DenseNet）
- **run_train.py**：单机单实验训练入口
- **run_parallel.py**：多配置、多 GPU 并行（JSON 或默认列表）
- **run_batch_all.py**：**批量跑全量 模型×数据集 实验**
- **pytorch_cifar10_main.py**：旧版单文件脚本（保留兼容）
- **infer.py**：权重分布可视化

### 运行方式

#### 1. 单次训练

```bash
python run_train.py --model_name ProbGoogLeNet --dataset CIFAR10 --epochs 100 --gpu_id 0
# 换数据集示例
python run_train.py --model_name VGG16 --dataset CIFAR100 --epochs 50
python run_train.py --model_name LeNet5 --dataset MNIST
```

结果会落在 `outputs/<时间戳>_<实验名>/` 下：

- `checkpoint/best.pth`、`last.pth`
- `logs/train.log`、TensorBoard 事件文件
- `metrics.csv`、`accuracy.png`

#### 2. 实时查看 loss 曲线（TensorBoard）

```bash
tensorboard --logdir=outputs --port=6006
```

浏览器打开 `http://localhost:6006`，在 SCALARS 里查看 loss/accuracy 曲线。

若打开 TensorBoard 后显示 "No dashboards are active"，用绝对路径指定日志目录：

```bash
tensorboard --logdir=/path/to/project/outputs --port=6006
```

#### 3. 多配置、多 GPU 并行

编辑 `run_parallel.py` 中的 `get_default_experiments()`，或使用 JSON 文件（每行一个实验配置）：

```bash
python run_parallel.py --experiments config/experiments_example.json --output_root outputs
```

#### 4. 批量跑全量实验

打开 **`config/batch_run_config.py`**，按需修改：

- **MAX_WORKERS**：并行数量
- **MODELS**：要跑的模型列表，`None` 表示全部
- **DATASETS**：要跑的数据集列表，`None` 表示全部
- **GPU_IDS**：使用的 GPU 编号列表

```bash
python run_batch_all.py
```

也可用命令行覆盖：`--models`、`--datasets`、`--gpu_ids`、`--max_workers`、`--epochs` 等。

#### 5. 推理/权重可视化

```bash
python infer.py --checkpoint outputs/某次实验/checkpoint/best.pth --model ProbResNet18
```

### 依赖

- Python 3.7+
- PyTorch、torchvision
- tensorboard、pandas、matplotlib

安装示例：`pip install torch torchvision tensorboard pandas matplotlib`

---

## English

A PyTorch implementation of probabilistic computing networks (PbitNet), using the **tanh × sigmoid composite activation** (`tanh(x)*sigmoid(x)`) and validated across multiple datasets.

### Activation Comparison (Prob* models only)

Configure in **config/batch_run_config.py**:
- **ACTIVATION_CHOICES**: available activation list (relu, sigmoid, tanh, **tanh_sigmoid**, silu, gelu, leaky_relu)
- **ACTIVATIONS**: `None` = uses default tanh_sigmoid only; set to a list like `["relu", "tanh_sigmoid", "silu"]` to run batch experiments for each Prob* model × dataset × activation combination

Single training with a specific activation: `python run_train.py --model_name ProbResNet18 --activation relu`

### Supported Models (config.MODEL_CHOICES)

- **Prob series**: ProbGoogLeNet, ProbResNet18/34/50/101/152 (with tanh*sigmoid + BinarySTE)
- **Classic CNNs**: AlexNet, VGG11/13/16/19, LeNet5/LeNet3, DenseNet121/169/201/161

### Supported Datasets

- CIFAR10, CIFAR100, MNIST, FashionMNIST

### Project Structure

- **config/**: unified parameter configuration (MODEL_CHOICES, DATASET_CHOICES, ExperimentConfig)
- **core/**: core logic (train_loop, datasets, logger, run_dir)
- **backbones/**: network architectures (Prob*, AlexNet, VGG, SimpleCNN, DenseNet)
- **run_train.py**: single-experiment training entry point
- **run_parallel.py**: multi-config, multi-GPU parallel execution (JSON or default lists)
- **run_batch_all.py**: **batch run of all model × dataset combinations**
- **pytorch_cifar10_main.py**: legacy single-file script (kept for compatibility)
- **infer.py**: weight distribution visualization

### Usage

#### 1. Single Training

```bash
python run_train.py --model_name ProbGoogLeNet --dataset CIFAR10 --epochs 100 --gpu_id 0
# Different datasets
python run_train.py --model_name VGG16 --dataset CIFAR100 --epochs 50
python run_train.py --model_name LeNet5 --dataset MNIST
```

Results are saved under `outputs/<timestamp>_<experiment_name>/`:

- `checkpoint/best.pth`, `last.pth`
- `logs/train.log`, TensorBoard event files
- `metrics.csv`, `accuracy.png`

#### 2. Real-time Loss Curves (TensorBoard)

```bash
tensorboard --logdir=outputs --port=6006
```

Open `http://localhost:6006` in a browser and view loss/accuracy curves under SCALARS.

If TensorBoard shows "No dashboards are active", use an absolute path:

```bash
tensorboard --logdir=/path/to/project/outputs --port=6006
```

#### 3. Multi-Config, Multi-GPU Parallel

Edit `get_default_experiments()` in `run_parallel.py`, or use a JSON file:

```bash
python run_parallel.py --experiments config/experiments_example.json --output_root outputs
```

#### 4. Batch Run All Experiments

Open **`config/batch_run_config.py`** and adjust:

- **MAX_WORKERS**: number of concurrent experiments
- **MODELS**: list of models, `None` for all
- **DATASETS**: list of datasets, `None` for all
- **GPU_IDS**: list of GPU indices

```bash
python run_batch_all.py
```

CLI overrides: `--models`, `--datasets`, `--gpu_ids`, `--max_workers`, `--epochs`, etc.

#### 5. Inference / Weight Visualization

```bash
python infer.py --checkpoint outputs/<experiment>/checkpoint/best.pth --model ProbResNet18
```

### Dependencies

- Python 3.7+
- PyTorch, torchvision
- tensorboard, pandas, matplotlib

Install: `pip install torch torchvision tensorboard pandas matplotlib`
