# PbitNet — Probabilistic Binary Neural Networks in PyTorch

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
