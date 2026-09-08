"""
Phase1 U-Net Generator (v1): RGB (3ch) → NIR (1ch) — 4层
=========================================================
轻量级 4 层 U-Net，用于 Pix2Pix 框架的 RGB→NIR 图像翻译。
此版本为早期训练版本，checkpoint保存在 checkpoints/phase1/ 下。

架构:
    编码器 (4层): 3→64→128→256→512
    瓶颈层: 512→512
    解码器 (4层): 512→512→256→128→64
    输出头: ConvTranspose→Tanh

支持输入尺寸: 256×256 / 512×512 等 32 的倍数。
"""

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """4层 U-Net 生成器: RGB → NIR

    Args:
        in_channels: 输入通道数 (默认 3, RGB)
        out_channels: 输出通道数 (默认 1, NIR)
        base_filters: 编码器首层特征数 (默认 64, 逐层翻倍)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters

        self.enc1 = UNetDownBlock(in_channels, f, normalize=False)
        self.enc2 = UNetDownBlock(f, f * 2)
        self.enc3 = UNetDownBlock(f * 2, f * 4)
        self.enc4 = UNetDownBlock(f * 4, f * 8)

        self.bottleneck = UNetDownBlock(f * 8, f * 8)

        self.dec4 = UNetUpBlock(f * 8, f * 8, dropout=0.5)
        self.dec3 = UNetUpBlock(f * 16, f * 4, dropout=0.5)
        self.dec2 = UNetUpBlock(f * 8, f * 2)
        self.dec1 = UNetUpBlock(f * 4, f)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.output(d1)
