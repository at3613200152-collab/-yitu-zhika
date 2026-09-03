"""
PatchDiscriminator: 70×70 PatchGAN 判别器 (Phase1)
====================================================
用于 RGB→NIR 生成器训练的条件判别器。

架构:
    4 层 Conv→BatchNorm→LeakyReLU + 1 层输出 Conv (1 通道)

    Layer 1: Conv(in→64, k4, s2, p1) → BN → LeakyReLU(0.2)     # 下采样
    Layer 2: Conv(64→128, k4, s2, p1) → BN → LeakyReLU(0.2)    # 下采样
    Layer 3: Conv(128→256, k4, s2, p1) → BN → LeakyReLU(0.2)   # 下采样
    Layer 4: Conv(256→512, k4, s1, p1) → BN → LeakyReLU(0.2)   # 保持分辨率
    Output:  Conv(512→1, k4, s1, p1)                            # 1 通道 logits

    输入: concat(RGB, NIR) = 4 通道 (条件 GAN)
    输出: 1 通道 logits 图 (配合 BCEWithLogitsLoss 使用)

    感受野: 70×70

权重初始化: Conv Normal(0, 0.02), BN weight=1 bias=0
参考: Isola et al., CVPR 2017
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN 条件判别器。

    Args:
        in_channels: 输入通道数 (RGB 3 + NIR 1 = 4)
        ndf: 首层卷积通道数 (默认 64, 逐层翻倍)
    """

    def __init__(self, in_channels: int = 4, ndf: int = 64):
        super().__init__()

        kw = 4   # kernel size
        pad = 1  # padding

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kw, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kw, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kw, stride=1, padding=pad, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, nir], dim=1)  # [B, 4, H, W]
        return self.model(x)
