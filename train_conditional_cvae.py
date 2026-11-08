import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


PROJECT_ROOT = Path(__file__).resolve().parent

# 设置 matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 配置参数
# =========================


class Config:
    image_size = 28
    image_channels = 1
    image_dim = image_size * image_size

    latent_dim = 64
    hidden_dim = 512
    num_classes = 10
    label_emb_dim = 10

    batch_size = 128
    epochs = 20
    lr = 1e-3
    beta = 1.0

    save_every = 5
    sample_every = 1

    save_dir = PROJECT_ROOT / "cvae_conditional_results"
    digit_output_dir = PROJECT_ROOT / "cvae_generated_digits_output"
    model_name = "conditional_cvae_mnist.pth"
    loss_curve_name = "cvae_loss_curve.png"
    sample_grid_name = "cvae_training_samples.png"

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
os.makedirs(cfg.digit_output_dir, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2. 数据加载
# =========================


def get_dataloader(num_workers=0):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        transform=transform,
        download=True,
    )

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
# 3. 条件 CVAE 网络
# =========================


class ConditionalCVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(cfg.num_classes, cfg.label_emb_dim)

        encoder_input_dim = cfg.image_dim + cfg.label_emb_dim
        decoder_input_dim = cfg.latent_dim + cfg.label_emb_dim

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(cfg.hidden_dim // 2, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim // 2, cfg.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, cfg.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.image_dim),
            nn.Tanh(),
        )

    def encode(self, images, labels):
        image_features = images.view(images.size(0), -1)
        label_features = self.label_emb(labels)
        x = torch.cat([image_features, label_features], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label_features = self.label_emb(labels)
        x = torch.cat([z, label_features], dim=1)
        images = self.decoder(x)
        return images.view(-1, cfg.image_channels, cfg.image_size, cfg.image_size)

    def forward(self, images, labels):
        mu, logvar = self.encode(images, labels)
        z = self.reparameterize(mu, logvar)
        recon_images = self.decode(z, labels)
        return recon_images, mu, logvar


# =========================
# 4. 训练与评估
# =========================


def cvae_loss(recon_images, real_images, mu, logvar):
    recon_loss = F.mse_loss(recon_images, real_images, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + cfg.beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0

    for images, labels in dataloader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        recon_images, mu, logvar = model(images, labels)
        total_loss, recon_loss, kl_loss = cvae_loss(recon_images, images, mu, logvar)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()

    avg_total_loss = total_loss_sum / len(dataloader)
    avg_recon_loss = recon_loss_sum / len(dataloader)
    avg_kl_loss = kl_loss_sum / len(dataloader)

    print(
        f"第 {epoch}/{cfg.epochs} 轮训练完成，"
        f"Total Loss: {avg_total_loss:.6f}，"
        f"Recon Loss: {avg_recon_loss:.6f}，"
        f"KL Loss: {avg_kl_loss:.6f}"
    )

    return avg_total_loss, avg_recon_loss, avg_kl_loss


@torch.no_grad()
def evaluate_cvae(model, dataloader):
    model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0

    for images, labels in dataloader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        recon_images, mu, logvar = model(images, labels)
        total_loss, recon_loss, kl_loss = cvae_loss(recon_images, images, mu, logvar)

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()

    return (
        total_loss_sum / len(dataloader),
        recon_loss_sum / len(dataloader),
        kl_loss_sum / len(dataloader),
    )


# =========================
# 5. 结果保存
# =========================


@torch.no_grad()
def generate_by_digit(model, num_images_per_digit=16):
    model.eval()

    for digit in range(cfg.num_classes):
        labels = torch.full(
            (num_images_per_digit,), digit, device=cfg.device, dtype=torch.long
        )
        z = torch.randn(num_images_per_digit, cfg.latent_dim, device=cfg.device)
        samples = model.decode(z, labels)

        samples = samples.clamp(-1, 1)
        samples = (samples + 1) / 2

        grid = make_grid(samples, nrow=4, padding=2)
        save_path = os.path.join(cfg.digit_output_dir, f"cvae_generated_digit_{digit}.png")
        save_image(grid, save_path)

        print(f"数字 {digit} 的 CVAE 生成结果已保存到: {save_path}")


@torch.no_grad()
def save_training_sample_grid(model, epoch):
    model.eval()

    labels = torch.arange(cfg.num_classes, device=cfg.device).repeat_interleave(10)
    z = torch.randn(labels.size(0), cfg.latent_dim, device=cfg.device)
    samples = model.decode(z, labels)

    samples = samples.clamp(-1, 1)
    samples = (samples + 1) / 2

    grid = make_grid(samples, nrow=10, padding=2)
    save_path = os.path.join(cfg.save_dir, f"epoch_{epoch}_samples.png")
    save_image(grid, save_path)


def save_loss_curve(train_losses, test_losses):
    save_path = os.path.join(cfg.save_dir, cfg.loss_curve_name)
    epochs = range(1, len(train_losses["total"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses["total"], label="训练 Total Loss")
    plt.plot(epochs, train_losses["recon"], label="训练 Recon Loss")
    plt.plot(epochs, train_losses["kl"], label="训练 KL Loss")
    plt.plot(epochs, test_losses["total"], "--", label="测试 Total Loss")
    plt.plot(epochs, test_losses["recon"], "--", label="测试 Recon Loss")
    plt.plot(epochs, test_losses["kl"], "--", label="测试 KL Loss")
    plt.xlabel("训练轮数")
    plt.ylabel("Loss")
    plt.title("Conditional CVAE 训练集与测试集 Loss 曲线")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"CVAE Loss 曲线已保存到: {save_path}")


def save_checkpoint(model, optimizer, epoch, losses, filename):
    save_path = os.path.join(cfg.save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "config": {
                "image_size": cfg.image_size,
                "image_channels": cfg.image_channels,
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "num_classes": cfg.num_classes,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "beta": cfg.beta,
            },
        },
        save_path,
    )

    print(f"CVAE 模型已保存到: {save_path}")


def load_model(checkpoint_path):
    model = ConditionalCVAE().to(cfg.device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# =========================
# 6. 训练主函数
# =========================


def train():
    set_seed(cfg.seed)

    train_loader, test_loader = get_dataloader(num_workers=0)

    model = ConditionalCVAE().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses = {"total": [], "recon": [], "kl": []}
    test_losses = {"total": [], "recon": [], "kl": []}

    print(f"当前使用设备: {cfg.device}")
    print("开始训练 Conditional CVAE 对比模型...")

    for epoch in range(1, cfg.epochs + 1):
        train_total_loss, train_recon_loss, train_kl_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
        )

        test_total_loss, test_recon_loss, test_kl_loss = evaluate_cvae(
            model=model,
            dataloader=test_loader,
        )

        train_losses["total"].append(train_total_loss)
        train_losses["recon"].append(train_recon_loss)
        train_losses["kl"].append(train_kl_loss)
        test_losses["total"].append(test_total_loss)
        test_losses["recon"].append(test_recon_loss)
        test_losses["kl"].append(test_kl_loss)

        print(
            f"第 {epoch}/{cfg.epochs} 轮结果："
            f"训练 Total Loss = {train_total_loss:.6f}，"
            f"训练 Recon Loss = {train_recon_loss:.6f}，"
            f"训练 KL Loss = {train_kl_loss:.6f}，"
            f"测试 Total Loss = {test_total_loss:.6f}，"
            f"测试 Recon Loss = {test_recon_loss:.6f}，"
            f"测试 KL Loss = {test_kl_loss:.6f}"
        )

        if epoch % cfg.sample_every == 0:
            save_training_sample_grid(model, epoch)

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                losses={"train_losses": train_losses, "test_losses": test_losses},
                filename=f"epoch_{epoch}_{cfg.model_name}",
            )

    losses = {"train_losses": train_losses, "test_losses": test_losses}

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        losses=losses,
        filename=cfg.model_name,
    )

    save_loss_curve(train_losses=train_losses, test_losses=test_losses)
    generate_by_digit(model, num_images_per_digit=16)

    print("Conditional CVAE 对比实验完成。")
    return model, losses


def generate_only(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_dir, cfg.model_name)

    if not os.path.exists(checkpoint_path):
        print(f"未找到模型文件: {checkpoint_path}")
        return

    model = load_model(checkpoint_path)
    generate_by_digit(model, num_images_per_digit=16)


if __name__ == "__main__":
    train()
