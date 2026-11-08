import os
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

# 从训练脚本中复用模型与配置、扩散实现
from train_conditional_ddpm import cfg, ConditionalUNet, Diffusion

PROJECT_ROOT = Path(__file__).resolve().parent

# 本脚本输出目录，每个数字保存一张 4x4 网格图
GENERATE_OUTPUT_DIR = PROJECT_ROOT / "generated_digits_output"
os.makedirs(GENERATE_OUTPUT_DIR, exist_ok=True)


def load_model(checkpoint_path):
    model = ConditionalUNet(
        image_channels=cfg.image_channels,
        base_channels=64,
        time_emb_dim=128,
        num_classes=cfg.num_classes,
    ).to(cfg.device)

    # 优先使用 weights_only=True 以避免在反序列化时执行任意的 pickle 代码。
    # 如果当前 PyTorch 版本不支持该参数，则回退到旧的 torch.load 调用。
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=cfg.device, weights_only=True
        )
    except TypeError:
        # Older PyTorch doesn't support weights_only kwarg
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded checkpoint with model_state_dict.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict.")

    model.eval()
    return model


@torch.no_grad()
def generate_all_digits(num_images_per_digit=16):
    checkpoint_path = os.path.join(cfg.save_dir, cfg.model_name)

    if not os.path.exists(checkpoint_path):
        print(f"未找到模型文件: {checkpoint_path}")
        return

    model = load_model(checkpoint_path)

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=cfg.device,
    )

    for digit in range(cfg.num_classes):
        labels = torch.full(
            (num_images_per_digit,),
            fill_value=digit,
            device=cfg.device,
            dtype=torch.long,
        )

        samples = diffusion.sample(
            model=model,
            labels=labels,
            image_size=cfg.image_size,
            channels=cfg.image_channels,
        )

        # 训练时用了 Normalize((0.5,), (0.5,))，所以这里要反归一化
        samples = samples.clamp(-1, 1)
        samples = (samples + 1) / 2

        grid = make_grid(samples, nrow=4, padding=2)

        save_path = os.path.join(GENERATE_OUTPUT_DIR, f"generated_digit_{digit}.png")

        save_image(grid, save_path)

        print(f"数字 {digit} 的生成结果已保存到: {save_path}")


def main():
    print(f"Using device: {cfg.device}")
    generate_all_digits(num_images_per_digit=16)


if __name__ == "__main__":
    main()
