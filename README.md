# Pytorch PbitNet

基于 PyTorch 的概率计算网络（PbitNet），使用 **tanh 与 sigmoid 复合激活函数**（`tanh(x)*sigmoid(x)`）在多数据集上验证性能。

## 激活函数对比（仅 Prob* 模型）

在 **config/batch_run_config.py** 中可配置：
- **ACTIVATION_CHOICES**：可选激活列表（relu, sigmoid, tanh, **tanh_sigmoid**, silu, gelu, leaky_relu）
- **ACTIVATIONS**：`None` = 仅用默认 tanh_sigmoid；设为列表如 `["relu", "tanh_sigmoid", "silu"]` 时，批量跑会对每个 Prob* 模型 × 数据集 × 激活 各跑一次，便于对比

单次训练也可指定：`python run_train.py --model_name ProbResNet18 --activation relu`

## 支持的模型（config.MODEL_CHOICES）

- **Prob 系列**：ProbGoogLeNet，ProbResNet18/34/50/101/152（含 tanh*sigmoid + BinarySTE）
- **经典 CNN**：AlexNet，VGG11/13/16/19，LeNet5/LeNet3，DenseNet121/169/201/161

## 支持的数据集

- CIFAR10、CIFAR100、MNIST、FashionMNIST

## 项目结构

- **config/**：统一参数配置（MODEL_CHOICES、DATASET_CHOICES、ExperimentConfig）
- **core/**：核心逻辑（train_loop、datasets、logger、run_dir）
- **backbones/**：网络结构（Prob*、AlexNet、VGG、SimpleCNN、DenseNet）
- **run_train.py**：单机单实验训练入口
- **run_parallel.py**：多配置、多 GPU 并行（JSON 或默认列表）
- **run_batch_all.py**：**批量跑全量 模型×数据集 实验**
- **pytorch_cifar10_main.py**：旧版单文件脚本（保留兼容）
- **infer.py**：权重分布可视化

## 运行方式

### 1. 单次训练

```bash
python run_train.py --model_name ProbGoogLeNet --dataset CIFAR10 --epochs 100 --gpu_id 0
# 换数据集示例
python run_train.py --model_name VGG16 --dataset CIFAR100 --epochs 50
python run_train.py --model_name LeNet5 --dataset MNIST
```

结果会落在 `outputs/<时间戳>_<实验名>/` 下：

- `checkpoint/best.pth`、`last.pth`
- `logs/train.log`、TensorBoard 事件文件（可用 `tensorboard --logdir=outputs` 实时看 loss/acc）
- `metrics.csv`、`accuracy.png`

### 2. 实时查看 loss 曲线（TensorBoard）

**若打开 TensorBoard 后显示 “No dashboards are active”：**

- 表示当前 `outputs/` 下还没有任何训练写入的日志。可先生成**示例数据**再打开 TensorBoard：
  ```bash
  conda activate pytorch   # 或你的训练环境
  python scripts/write_tb_demo.py
  tensorboard --logdir=outputs --port=6006
  ```
- 或用**绝对路径**指定日志目录（避免因当前工作目录不同而读不到）：
  ```bash
  tensorboard --logdir=/home/probclac/Data/Project_public/Pytorch_PbitNet-by_curor/outputs --port=6006
  ```

正常训练时，`run_train.py` / `run_batch_all.py` 会把曲线写入 `outputs/<时间戳>_<实验名>/logs/`，TensorBoard 会递归扫描 `outputs/` 下所有 event 文件。

```bash
tensorboard --logdir=outputs --port=6006
```

浏览器打开 `http://localhost:6006`，在 SCALARS 里查看 loss/accuracy 曲线。

### 3. 多配置、多 GPU 并行

编辑 `run_parallel.py` 中的 `get_default_experiments()`，或使用 JSON 文件（每行一个实验配置）：

```bash
python run_parallel.py --experiments config/experiments_example.json --output_root outputs
```

每个实验会绑定到对应 `gpu_id` 的显卡；若只有单卡，可将所有实验的 `gpu_id` 设为 0（将顺序执行）。

### 4. 批量跑全量实验（所有模型 × 所有数据集）

**推荐**：改配置文件后直接运行，不用记命令行参数。

打开 **`config/batch_run_config.py`**，按需修改：

- **MAX_WORKERS**：并行数量（同时跑几个实验）
- **MODELS**：要跑的模型列表，`None` 表示全部
- **DATASETS**：要跑的数据集列表，`None` 表示全部
- **GPU_IDS**：使用的 GPU 编号列表

然后执行：

```bash
python run_batch_all.py
```

也可用命令行覆盖：`--models`、`--datasets`、`--gpu_ids`、`--max_workers`、`--epochs` 等。

### 5. 推理/权重可视化

```bash
python infer.py --checkpoint outputs/某次实验/checkpoint/best.pth --model ProbResNet18
```

## 依赖

- Python 3.7+
- PyTorch、torchvision
- tensorboard、pandas、matplotlib

安装示例：`pip install torch torchvision tensorboard pandas matplotlib`
