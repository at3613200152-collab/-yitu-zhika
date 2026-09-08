"""
U-Net 生成器
=============
用于Pix2Pix框架的生成器网络，将RGB图像转换为NIR图像。

架构:
    编码器: 64→128→256→512→512, 每层 Conv+BN+LeakyReLU
    解码器: 对称上采样+skip connection
    输入: RGB 3通道
    输出: NIR 1通道

参考: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017
"""

import torch
import torch.nn as nn
from typing import List, Optional


class UNetDownBlock(nn.Module):
    """U-Net下采样块: Conv → BN → LeakyReLU
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        normalize: 是否使用BatchNorm (最底层不需要)
    """

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUpBlock(nn.Module):
    """U-Net上采样块: ConvTranspose → BN → Dropout → ReLU → 拼接skip connection
    
    Args:
        in_channels: 输入通道数（含skip connection通道）
        out_channels: 输出通道数
        dropout: dropout概率
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        # 拼接skip connection (沿通道维度)
        x = torch.cat([x, skip_input], dim=1)
        return x


class UNetGenerator(nn.Module):
    """U-Net生成器网络

    将RGB图像转换为NIR图像。

    Args:
        input_channels: 输入通道数 (默认3, RGB)
        output_channels: 输出通道数 (默认1, NIR)
        base_features: 编码器首层特征数 (默认64)
        use_dropout: 解码器是否使用dropout (默认True, 训练时增加多样性)
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_features: int = 64,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        f = base_features  # 64
        dropout = 0.5 if use_dropout else 0.0

        # ==================== 编码器 ====================
        # 输入: [input_channels, 256, 256]
        self.down1 = UNetDownBlock(input_channels, f, normalize=False)       # → [64, 128, 128]
        self.down2 = UNetDownBlock(f, f * 2)                                  # → [128, 64, 64]
        self.down3 = UNetDownBlock(f * 2, f * 4)                              # → [256, 32, 32]
        self.down4 = UNetDownBlock(f * 4, f * 8)                              # → [512, 16, 16]
        self.down5 = UNetDownBlock(f * 8, f * 8)                              # → [512, 8, 8]
        self.down6 = UNetDownBlock(f * 8, f * 8)                              # → [512, 4, 4]
        self.down7 = UNetDownBlock(f * 8, f * 8)                              # → [512, 2, 2]

        # 最底层: 不使用BN
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )  # → [512, 1, 1]

        # ==================== 解码器 ====================
        self.up1 = UNetUpBlock(f * 8, f * 8, dropout=dropout)                # ← [512, 2, 2]
        self.up2 = UNetUpBlock(f * 16, f * 8, dropout=dropout)               # ← [512, 4, 4]
        self.up3 = UNetUpBlock(f * 16, f * 8, dropout=dropout)               # ← [512, 8, 8]
        self.up4 = UNetUpBlock(f * 16, f * 8)                                 # ← [512, 16, 16]
        self.up5 = UNetUpBlock(f * 16, f * 4)                                 # ← [256, 32, 32]
        self.up6 = UNetUpBlock(f * 8, f * 2)                                  # ← [128, 64, 64]
        self.up7 = UNetUpBlock(f * 4, f)                                      # ← [64, 128, 128]

        # 最终输出层
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )  # → [output_channels, 256, 256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入RGB图像 [B, 3, 256, 256]

        Returns:
            预测的NIR图像 [B, 1, 256, 256], 值域[-1, 1]
        """
        # 编码器 - 保存每层特征用于skip connection
        d1 = self.down1(x)       # [B, 64, 128, 128]
        d2 = self.down2(d1)      # [B, 128, 64, 64]
        d3 = self.down3(d2)      # [B, 256, 32, 32]
        d4 = self.down4(d3)      # [B, 512, 16, 16]
        d5 = self.down5(d4)      # [B, 512, 8, 8]
        d6 = self.down6(d5)      # [B, 512, 4, 4]
        d7 = self.down7(d6)      # [B, 512, 2, 2]

        # 瓶颈层
        bottleneck = self.bottleneck(d7)  # [B, 512, 1, 1]

        # 解码器 - 拼接skip connection
        u1 = self.up1(bottleneck, d7)  # [B, 1024, 2, 2]
        u2 = self.up2(u1, d6)          # [B, 1024, 4, 4]
        u3 = self.up3(u2, d5)          # [B, 1024, 8, 8]
        u4 = self.up4(u3, d4)          # [B, 1024, 16, 16]
        u5 = self.up5(u4, d3)          # [B, 512, 32, 32]
        u6 = self.up6(u5, d2)          # [B, 256, 64, 64]
        u7 = self.up7(u6, d1)          # [B, 128, 128, 128]

        # 最终输出
        output = self.final(u7)  # [B, output_channels, 256, 256]

        return output


if __name__ == "__main__":
    # 测试U-Net生成器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = UNetGenerator(
        input_channels=3,
        output_channels=1,
        base_features=64,
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"输出范围: [{y.min().item():.3f}, {y.max().item():.3f}]")
