import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 常用黑体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent

# =========================
# 1. 配置参数
# =========================


class Config:
    image_size = 28
    image_channels = 1

    batch_size = 128
    epochs = 20

    lr = 1e-3

    timesteps = 300
    beta_start = 1e-4
    beta_end = 0.02

    num_classes = 10

    save_dir = PROJECT_ROOT / "ddpm_conditional_results"
    model_name = "conditional_ddpm_mnist.pth"

    # 每隔多少轮保存一次中间模型
    save_every = 5

    # Loss 曲线图片名称
    loss_curve_name = "loss_curve.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)


# =========================
# 2. 数据加载
# =========================


def get_dataloader(num_workers=0):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # 训练集
    train_dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        transform=transform,
        download=True,
    )

    # 测试集
    test_dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, test_loader


# =========================
# 3. 时间步嵌入
# =========================


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb


# =========================
# 4. 卷积模块
# =========================


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.emb_layer = nn.Linear(emb_dim, out_channels)

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        emb_out = self.emb_layer(emb)
        emb_out = emb_out[:, :, None, None]

        h = h + emb_out

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h


# =========================
# 5. 条件 U-Net 模型
# =========================


class ConditionalUNet(nn.Module):
    def __init__(
        self, image_channels=1, base_channels=64, time_emb_dim=128, num_classes=10
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.down1 = ConvBlock(image_channels, base_channels, time_emb_dim)

        self.down2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        self.middle = ConvBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.up1 = ConvBlock(base_channels * 3, base_channels, time_emb_dim)

        self.out = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x, t, y):
        # 时间步嵌入
        t_emb = self.time_mlp(t)

        # 类别标签嵌入
        y_emb = self.label_emb(y)

        # 条件信息融合
        emb = t_emb + y_emb

        # 下采样
        x1 = self.down1(x, emb)

        x2 = self.pool(x1)
        x2 = self.down2(x2, emb)

        # 中间层
        mid = self.middle(x2, emb)

        # 上采样并拼接跳跃连接
        up = self.up(mid)
        up = torch.cat([up, x1], dim=1)

        up = self.up1(up, emb)

        return self.out(up)


# =========================
# 6. 扩散过程
# =========================


class Diffusion:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device

        # 线性 beta 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        self.alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def extract(self, values, t, x_shape):
        batch_size = t.shape[0]

        out = values[t]

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：根据时间步 t 给原图 x_start 添加噪声，得到 x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)

        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, y):
        """
        反向去噪过程中的单步采样：x_t -> x_{t-1}
        """
        betas_t = self.extract(self.betas, t, x.shape)

        sqrt_one_minus_alpha_bar_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        pred_noise = model(x, t, y)

        model_mean = sqrt_recip_alpha_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alpha_bar_t
        )

        if t[0].item() == 0:
            return model_mean

        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)

        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, labels, image_size, channels=1):
        """
        完整反向采样过程：从随机噪声生成图像
        """
        model.eval()

        batch_size = labels.shape[0]

        x = torch.randn(
            batch_size, channels, image_size, image_size, device=self.device
        )

        for i in range(self.timesteps - 1, -1, -1):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            x = self.p_sample(model, x, t, labels)

        return x


# =========================
# 7. 单轮训练函数
# =========================


def train_one_epoch(model, diffusion, dataloader, optimizer, epoch):
    model.train()

    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        batch_size = x.shape[0]

        # 随机采样时间步 t
        t = torch.randint(
            low=0, high=cfg.timesteps, size=(batch_size,), device=cfg.device
        ).long()

        # 生成真实噪声
        noise = torch.randn_like(x)

        # 前向扩散：x_0 -> x_t
        x_t = diffusion.q_sample(x_start=x, t=t, noise=noise)

        # 模型预测噪声
        pred_noise = model(x_t, t, y)

        # DDPM 标准训练目标：预测加入的噪声
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"第 {epoch}/{cfg.epochs} 轮训练完成，" f"训练 Loss: {avg_loss:.6f}")

    return avg_loss


# =========================
# 8. 测试 Loss 计算函数
# =========================


@torch.no_grad()
def evaluate_loss(model, diffusion, dataloader):
    model.eval()

    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        batch_size = x.shape[0]

        # 测试时同样随机采样时间步
        t = torch.randint(
            low=0, high=cfg.timesteps, size=(batch_size,), device=cfg.device
        ).long()

        # 生成测试噪声
        noise = torch.randn_like(x)

        # 对测试图像加噪
        x_t = diffusion.q_sample(x_start=x, t=t, noise=noise)

        # 模型预测噪声
        pred_noise = model(x_t, t, y)

        # 计算测试集上的噪声预测损失
        loss = F.mse_loss(pred_noise, noise)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss


# =========================
# 9. 保存 Loss 曲线
# =========================


def save_loss_curve(train_losses, test_losses):
    save_path = os.path.join(cfg.save_dir, cfg.loss_curve_name)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, label="训练 Loss")

    plt.plot(epochs, test_losses, label="测试 Loss")

    plt.xlabel("训练轮数")
    plt.ylabel("MSE Loss")
    plt.title("DDPM 训练集与测试集 Loss 曲线")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Loss 曲线已保存到: {save_path}")


# =========================
# 10. 保存模型
# =========================


def save_checkpoint(model, optimizer, epoch, train_losses, test_losses, filename):
    save_path = os.path.join(cfg.save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses,
            "config": {
                "image_size": cfg.image_size,
                "image_channels": cfg.image_channels,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "timesteps": cfg.timesteps,
                "beta_start": cfg.beta_start,
                "beta_end": cfg.beta_end,
                "num_classes": cfg.num_classes,
            },
        },
        save_path,
    )

    print(f"模型已保存到: {save_path}")


# =========================
# 11. 训练主函数
# =========================


def train():
    train_loader, test_loader = get_dataloader(num_workers=0)

    model = ConditionalUNet(
        image_channels=cfg.image_channels,
        base_channels=64,
        time_emb_dim=128,
        num_classes=cfg.num_classes,
    ).to(cfg.device)

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=cfg.device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses = []
    test_losses = []

    print(f"当前使用设备: {cfg.device}")
    print("开始训练条件 DDPM 模型...")

    for epoch in range(1, cfg.epochs + 1):
        # 训练一轮
        train_loss = train_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
        )

        # 每轮训练结束后，在测试集上计算一次 Loss
        test_loss = evaluate_loss(
            model=model, diffusion=diffusion, dataloader=test_loader
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"第 {epoch}/{cfg.epochs} 轮结果："
            f"训练 Loss = {train_loss:.6f}，"
            f"测试 Loss = {test_loss:.6f}"
        )

        # 保存中间模型
        if epoch % cfg.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=train_losses,
                test_losses=test_losses,
                filename=f"epoch_{epoch}_{cfg.model_name}",
            )

    # 保存最终模型
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        train_losses=train_losses,
        test_losses=test_losses,
        filename=cfg.model_name,
    )

    # 训练全部结束后，只保存一次 Loss 曲线
    save_loss_curve(train_losses=train_losses, test_losses=test_losses)

    print("Training finished.")

    return model, diffusion, train_losses, test_losses


# =========================
# 12. 主入口
# =========================

if __name__ == "__main__":
    train()
