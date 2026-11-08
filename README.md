# Conditional Digit Generation

基于条件扩散模型的手写数字生成

本项目使用 MNIST 数据集实现按标签生成手写数字的条件 DDPM，并提供 Conditional GAN 和 Conditional CVAE 作为对比模型。所有训练脚本均会自动下载 MNIST 数据集，并根据运行环境自动选择 CUDA 或 CPU。

数据集、模型权重和生成结果的路径均以脚本所在目录为根目录，因此可以从任意工作目录调用脚本，不会把运行产物写入其他位置。

## 文件说明

```text
.
├── train_conditional_ddpm.py    # 条件 DDPM：训练、测试损失评估与模型保存
├── generate_conditional_ddpm.py # 加载 DDPM 权重并生成 0-9 数字
├── train_conditional_gan.py     # Conditional GAN 对比实验
├── train_conditional_cvae.py    # Conditional CVAE 对比实验
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
```

## 环境要求

- Python 3.9 或更高版本
- PyTorch
- torchvision
- matplotlib
- 可选：支持 CUDA 的 NVIDIA GPU

安装依赖：

```bash
pip install -r requirements.txt
```

如需使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch。CPU 环境也可以运行，但训练和扩散采样耗时更长。

## 运行方法

### 1. 训练条件 DDPM

```bash
python train_conditional_ddpm.py
```

主要配置：

- 图像尺寸：28 x 28
- 批大小：128
- 训练轮数：20
- 扩散时间步：300
- 学习率：1e-3
- 噪声调度：线性 beta 调度，范围为 1e-4 至 0.02

训练结果保存在 `ddpm_conditional_results/`，包括最终权重、中间权重和损失曲线。

### 2. 使用 DDPM 生成数字

完成 DDPM 训练后运行：

```bash
python generate_conditional_ddpm.py
```

脚本默认读取：

```text
ddpm_conditional_results/conditional_ddpm_mnist.pth
```

生成的 0-9 数字网格图保存在 `generated_digits_output/`。每个类别生成 16 个样本，并排列为 4 x 4 网格。

### 3. 训练 Conditional GAN

```bash
python train_conditional_gan.py
```

模型权重、训练样本和损失曲线保存在 `gan_conditional_results/`，按类别生成的结果保存在 `gan_generated_digits_output/`。

### 4. 训练 Conditional CVAE

```bash
python train_conditional_cvae.py
```

模型权重、训练样本和损失曲线保存在 `cvae_conditional_results/`，按类别生成的结果保存在 `cvae_generated_digits_output/`。

## 数据集

四个训练流程均通过 `torchvision.datasets.MNIST` 读取数据。首次运行时，程序会将 MNIST 自动下载到 `data/`，因此提交包不包含数据集文件。

## 输出说明

模型权重、数据集、训练曲线和生成图片均属于运行产物，不包含在提交包中。重新运行对应脚本即可生成。输出目录和 `.pth` 文件已加入 `.gitignore`。

## 注意事项

- `generate_conditional_ddpm.py` 依赖 DDPM 训练生成的最终权重，必须先运行 `train_conditional_ddpm.py`。
- 训练轮数、批大小、学习率和保存路径可在各脚本的 `Config` 类中修改。
- GAN 与 CVAE 脚本用于对比实验，不是 DDPM 生成脚本的运行依赖。
- matplotlib 中文字体默认设置为 Windows 的 `SimHei`；其他系统若缺少该字体，只影响图表中的中文显示，不影响训练。
