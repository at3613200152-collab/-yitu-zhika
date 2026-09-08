"""
Pix2Pix 框架
=============
包含生成器和PatchGAN判别器的Pix2Pix条件GAN实现。

生成器: U-Net (来自unet.py)
判别器: 70x70 PatchGAN — 对图像的每个70x70局部区域判断真/假

损失函数:
    G_loss = L1_loss * λ_L1 + GAN_loss * λ_GAN
    D_loss = BCE_loss(D(real_pair), 1) + BCE_loss(D(fake_pair), 0)

参考: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .unet import UNetGenerator


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN 判别器

    判别器对输入图像的每个70x70局部区域输出一个真/假判断。
    输入为条件对: (RGB图像, NIR图像) 拼接为6通道。
    
    架构:
        C64 → C128 → C256 → C512 → C1
        其中 Ck = Conv-k-BN-LeakyReLU(0.2)
        首层不用BN，最后一层不用BN也不加激活

    Args:
        input_channels: 输入通道数 (RGB 3 + NIR 1 = 4)
        base_features: 首层特征数 (默认64)
    """

    def __init__(self, input_channels: int = 4, base_features: int = 64):
        super().__init__()
        f = base_features

        def discriminator_block(in_ch: int, out_ch: int, normalization: bool = True):
            """判别器基础块"""
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 输入: [input_channels, 256, 256]
            *discriminator_block(input_channels, f, normalization=False),      # → [64, 128, 128]
            *discriminator_block(f, f * 2),                                     # → [128, 64, 64]
            *discriminator_block(f * 2, f * 4),                                 # → [256, 32, 32]
            *discriminator_block(f * 4, f * 8),                                 # → [512, 16, 16]
            # 最后一层: stride=1
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),           # → [1, 15, 15]
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            img_a: 条件输入 (RGB图像) [B, 3, H, W]
            img_b: 目标/预测图像 (NIR图像) [B, 1, H, W]

        Returns:
            判别器输出 [B, 1, H', W'], 每个像素代表一个局部区域的真/假判断
        """
        # 在通道维度上拼接条件对
        x = torch.cat([img_a, img_b], dim=1)  # [B, 4, H, W]
        return self.model(x)


class Pix2PixModel(nn.Module):
    """Pix2Pix 完整模型（生成器+判别器）

    封装U-Net生成器和PatchGAN判别器，提供统一的训练接口。

    Args:
        input_channels: 生成器输入通道数 (默认3, RGB)
        output_channels: 生成器输出通道数 (默认1, NIR)
        base_features_g: 生成器基础特征数 (默认64)
        base_features_d: 判别器基础特征数 (默认64)
        lambda_l1: L1损失权重 (默认100.0)
        lambda_gan: GAN损失权重 (默认1.0)
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_features_g: int = 64,
        base_features_d: int = 64,
        lambda_l1: float = 100.0,
        lambda_gan: float = 1.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_gan = lambda_gan

        # 生成器
        self.generator = UNetGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            base_features=base_features_g,
        )

        # 判别器
        self.discriminator = PatchGANDiscriminator(
            input_channels=input_channels + output_channels,  # 拼接条件对
            base_features=base_features_d,
        )

        # 损失函数
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成器前向传播（推理时使用）

        Args:
            x: RGB输入图像 [B, 3, H, W]

        Returns:
            预测的NIR图像 [B, 1, H, W]
        """
        return self.generator(x)

    def compute_generator_loss(
        self,
        rgb: torch.Tensor,
        nir_real: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """计算生成器损失

        Args:
            rgb: RGB输入 [B, 3, H, W]
            nir_real: 真实NIR [B, 1, H, W]

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失明细
        """
        # 生成假NIR
        nir_fake = self.generator(rgb)

        # GAN损失: 让判别器认为假图是真图
        disc_fake = self.discriminator(rgb, nir_fake)
        # 创建与disc_fake同形状的全1标签
        valid = torch.ones_like(disc_fake)
        loss_gan = self.criterion_gan(disc_fake, valid)

        # L1重建损失
        loss_l1 = self.criterion_l1(nir_fake, nir_real)

        # 总损失
        total_loss = self.lambda_gan * loss_gan + self.lambda_l1 * loss_l1

        loss_dict = {
            "g_total": total_loss.item(),
            "g_gan": loss_gan.item(),
            "g_l1": loss_l1.item(),
        }

        return total_loss, loss_dict

    def compute_discriminator_loss(
        self,
        rgb: torch.Tensor,
        nir_real: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """计算判别器损失

        Args:
            rgb: RGB输入 [B, 3, H, W]
            nir_real: 真实NIR [B, 1, H, W]

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失明细
        """
        # 生成假NIR（不需要梯度）
        with torch.no_grad():
            nir_fake = self.generator(rgb)

        # 判别真图对
        disc_real = self.discriminator(rgb, nir_real)
        valid = torch.ones_like(disc_real)
        loss_real = self.criterion_gan(disc_real, valid)

        # 判别假图对
        disc_fake = self.discriminator(rgb, nir_fake.detach())
        fake = torch.zeros_like(disc_fake)
        loss_fake = self.criterion_gan(disc_fake, fake)

        # 总损失
        total_loss = (loss_real + loss_fake) * 0.5

        loss_dict = {
            "d_total": total_loss.item(),
            "d_real": loss_real.item(),
            "d_fake": loss_fake.item(),
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = Pix2PixModel(
        input_channels=3,
        output_channels=1,
        lambda_l1=100.0,
        lambda_gan=1.0,
    ).to(device)

    # 统计参数量
    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"生成器参数量: {g_params:,}")
    print(f"判别器参数量: {d_params:,}")

    # 测试前向传播
    rgb = torch.randn(2, 3, 256, 256).to(device)
    nir_real = torch.randn(2, 1, 256, 256).to(device)

    with torch.no_grad():
        nir_fake = model(rgb)
    print(f"生成器输出: {nir_fake.shape}")

    # 测试损失计算
    g_loss, g_dict = model.compute_generator_loss(rgb, nir_real)
    d_loss, d_dict = model.compute_discriminator_loss(rgb, nir_real)
    print(f"生成器损失: {g_dict}")
    print(f"判别器损失: {d_dict}")
