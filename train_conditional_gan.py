import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

    latent_dim = 100
    num_classes = 10
    label_emb_dim = 10

    batch_size = 128
    epochs = 20
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999

    save_every = 5
    sample_every = 1

    save_dir = PROJECT_ROOT / "gan_conditional_results"
    digit_output_dir = PROJECT_ROOT / "gan_generated_digits_output"
    model_name = "conditional_gan_mnist.pth"
    loss_curve_name = "gan_loss_curve.png"
    sample_grid_name = "gan_training_samples.png"

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
# 3. 条件 GAN 网络
# =========================


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(cfg.num_classes, cfg.label_emb_dim)

        input_dim = cfg.latent_dim + cfg.label_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, cfg.image_dim),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_features = self.label_emb(labels)
        x = torch.cat([z, label_features], dim=1)
        images = self.net(x)
        return images.view(-1, cfg.image_channels, cfg.image_size, cfg.image_size)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(cfg.num_classes, cfg.label_emb_dim)

        input_dim = cfg.image_dim + cfg.label_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, images, labels):
        image_features = images.view(images.size(0), -1)
        label_features = self.label_emb(labels)
        x = torch.cat([image_features, label_features], dim=1)
        return self.net(x)


# =========================
# 4. 训练与评估
# =========================


def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion, epoch):
    generator.train()
    discriminator.train()

    g_loss_total = 0.0
    d_loss_total = 0.0

    for real_images, labels in dataloader:
        real_images = real_images.to(cfg.device)
        labels = labels.to(cfg.device)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, 1, device=cfg.device)
        fake_targets = torch.zeros(batch_size, 1, device=cfg.device)

        # 训练判别器：真实图像判为真，生成图像判为假
        z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
        fake_images = generator(z, labels)

        real_logits = discriminator(real_images, labels)
        fake_logits = discriminator(fake_images.detach(), labels)

        d_real_loss = criterion(real_logits, real_targets)
        d_fake_loss = criterion(fake_logits, fake_targets)
        d_loss = d_real_loss + d_fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器：让判别器把生成图像判为真
        z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
        fake_images = generator(z, labels)
        fake_logits = discriminator(fake_images, labels)
        g_loss = criterion(fake_logits, real_targets)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    avg_g_loss = g_loss_total / len(dataloader)
    avg_d_loss = d_loss_total / len(dataloader)

    print(
        f"第 {epoch}/{cfg.epochs} 轮训练完成，"
        f"Generator Loss: {avg_g_loss:.6f}，"
        f"Discriminator Loss: {avg_d_loss:.6f}"
    )

    return avg_g_loss, avg_d_loss


@torch.no_grad()
def evaluate_gan(generator, discriminator, dataloader, criterion):
    generator.eval()
    discriminator.eval()

    g_loss_total = 0.0
    d_loss_total = 0.0

    for real_images, labels in dataloader:
        real_images = real_images.to(cfg.device)
        labels = labels.to(cfg.device)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, 1, device=cfg.device)
        fake_targets = torch.zeros(batch_size, 1, device=cfg.device)

        z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
        fake_images = generator(z, labels)

        real_logits = discriminator(real_images, labels)
        fake_logits = discriminator(fake_images, labels)

        d_loss = criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
        g_loss = criterion(fake_logits, real_targets)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    return g_loss_total / len(dataloader), d_loss_total / len(dataloader)


# =========================
# 5. 结果保存
# =========================


@torch.no_grad()
def generate_by_digit(generator, num_images_per_digit=16):
    generator.eval()

    for digit in range(cfg.num_classes):
        labels = torch.full(
            (num_images_per_digit,), digit, device=cfg.device, dtype=torch.long
        )
        z = torch.randn(num_images_per_digit, cfg.latent_dim, device=cfg.device)
        samples = generator(z, labels)

        samples = samples.clamp(-1, 1)
        samples = (samples + 1) / 2

        grid = make_grid(samples, nrow=4, padding=2)
        save_path = os.path.join(cfg.digit_output_dir, f"gan_generated_digit_{digit}.png")
        save_image(grid, save_path)

        print(f"数字 {digit} 的 GAN 生成结果已保存到: {save_path}")


@torch.no_grad()
def save_training_sample_grid(generator, epoch):
    generator.eval()

    labels = torch.arange(cfg.num_classes, device=cfg.device).repeat_interleave(10)
    z = torch.randn(labels.size(0), cfg.latent_dim, device=cfg.device)
    samples = generator(z, labels)

    samples = samples.clamp(-1, 1)
    samples = (samples + 1) / 2

    grid = make_grid(samples, nrow=10, padding=2)
    save_path = os.path.join(cfg.save_dir, f"epoch_{epoch}_samples.png")
    save_image(grid, save_path)


def save_loss_curve(train_g_losses, train_d_losses, test_g_losses, test_d_losses):
    save_path = os.path.join(cfg.save_dir, cfg.loss_curve_name)
    epochs = range(1, len(train_g_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_g_losses, label="训练 Generator Loss")
    plt.plot(epochs, train_d_losses, label="训练 Discriminator Loss")
    plt.plot(epochs, test_g_losses, "--", label="测试 Generator Loss")
    plt.plot(epochs, test_d_losses, "--", label="测试 Discriminator Loss")
    plt.xlabel("训练轮数")
    plt.ylabel("BCEWithLogits Loss")
    plt.title("Conditional GAN 训练集与测试集 Loss 曲线")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"GAN Loss 曲线已保存到: {save_path}")


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, losses, filename):
    save_path = os.path.join(cfg.save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "losses": losses,
            "config": {
                "image_size": cfg.image_size,
                "image_channels": cfg.image_channels,
                "latent_dim": cfg.latent_dim,
                "num_classes": cfg.num_classes,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
            },
        },
        save_path,
    )

    print(f"GAN 模型已保存到: {save_path}")


def load_generator(checkpoint_path):
    generator = Generator().to(cfg.device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


# =========================
# 6. 训练主函数
# =========================


def train():
    set_seed(cfg.seed)

    train_loader, test_loader = get_dataloader(num_workers=0)

    generator = Generator().to(cfg.device)
    discriminator = Discriminator().to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
    )

    train_g_losses = []
    train_d_losses = []
    test_g_losses = []
    test_d_losses = []

    print(f"当前使用设备: {cfg.device}")
    print("开始训练 Conditional GAN 对比模型...")

    for epoch in range(1, cfg.epochs + 1):
        train_g_loss, train_d_loss = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            criterion=criterion,
            epoch=epoch,
        )

        test_g_loss, test_d_loss = evaluate_gan(
            generator=generator,
            discriminator=discriminator,
            dataloader=test_loader,
            criterion=criterion,
        )

        train_g_losses.append(train_g_loss)
        train_d_losses.append(train_d_loss)
        test_g_losses.append(test_g_loss)
        test_d_losses.append(test_d_loss)

        print(
            f"第 {epoch}/{cfg.epochs} 轮结果："
            f"训练 G Loss = {train_g_loss:.6f}，"
            f"训练 D Loss = {train_d_loss:.6f}，"
            f"测试 G Loss = {test_g_loss:.6f}，"
            f"测试 D Loss = {test_d_loss:.6f}"
        )

        if epoch % cfg.sample_every == 0:
            save_training_sample_grid(generator, epoch)

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                epoch=epoch,
                losses={
                    "train_g_losses": train_g_losses,
                    "train_d_losses": train_d_losses,
                    "test_g_losses": test_g_losses,
                    "test_d_losses": test_d_losses,
                },
                filename=f"epoch_{epoch}_{cfg.model_name}",
            )

    losses = {
        "train_g_losses": train_g_losses,
        "train_d_losses": train_d_losses,
        "test_g_losses": test_g_losses,
        "test_d_losses": test_d_losses,
    }

    save_checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        epoch=cfg.epochs,
        losses=losses,
        filename=cfg.model_name,
    )

    save_loss_curve(
        train_g_losses=train_g_losses,
        train_d_losses=train_d_losses,
        test_g_losses=test_g_losses,
        test_d_losses=test_d_losses,
    )

    generate_by_digit(generator, num_images_per_digit=16)

    print("Conditional GAN 对比实验完成。")
    return generator, discriminator, losses


def generate_only(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.save_dir, cfg.model_name)

    if not os.path.exists(checkpoint_path):
        print(f"未找到模型文件: {checkpoint_path}")
        return

    generator = load_generator(checkpoint_path)
    generate_by_digit(generator, num_images_per_digit=16)


if __name__ == "__main__":
    train()
