"""
Phase1 U-Net Generator: RGB (3ch) → NIR (1ch)
==============================================
轻量级 4 层 U-Net，用于 Pix2Pix 框架的 RGB→NIR 图像翻译。

架构总览:
    编码器 (4 层下采样):
        Conv2d(stride=2) → BN → LeakyReLU(0.2)
        通道: 3 → 64 → 128 → 256 → 512

    瓶颈层:
        Conv2d(stride=2) → BN → LeakyReLU(0.2)
        通道: 512 → 512

    解码器 (4 层上采样 + skip connection):
        ConvTranspose2d(stride=2) → BN → ReLU
        通道: 512→512→256→128→64  (拼接 skip 后分别为 1024→512→256→128)

    输出头:
        ConvTranspose2d → Tanh    (输出范围 [-1, 1])
        通道: 128 → 1

支持输入尺寸: 256×256 / 512×512 等任意 32 的倍数。
权重初始化: Normal(mean=0, std=0.02)

参考: Isola et al., "Image-to-Image Translation with Conditional
      Adversarial Networks", CVPR 2017
"""

import torch
import torch.nn as nn


# ── 编码器块 ──────────────────────────────────────────────────────────────────
class UNetDownBlock(nn.Module):
    """下采样块: Conv2d(k=4, s=2, p=1) → [BN] → LeakyReLU(0.2)

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        normalize:    是否使用 BatchNorm (首层不用, 遵循 pix2pix 惯例)
    """

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── 解码器块 ──────────────────────────────────────────────────────────────────
class UNetUpBlock(nn.Module):
    """上采样块: ConvTranspose2d(k=4, s=2, p=1) → BN → [Dropout] → ReLU
    forward 时自动拼接 skip connection (沿通道维度)。

    Args:
        in_channels:  输入通道数 (不含 skip)
        out_channels: ConvTranspose 输出通道数
        dropout:      Dropout2d 概率, 0 表示不使用
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


# ── 生成器 ────────────────────────────────────────────────────────────────────
class UNetGenerator(nn.Module):
    """Phase1 U-Net 生成器: RGB → NIR

    Args:
        in_channels:  输入通道数 (默认 3, RGB)
        out_channels: 输出通道数 (默认 1, NIR)
        base_filters: 编码器首层特征数 (默认 64, 逐层翻倍)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
    ):
        super().__init__()
        f = base_filters  # 64

        # ── 编码器 (4 层下采样) ──────────────────────────────────────
        #   3  → 64   → 128  → 256  → 512
        self.enc1 = UNetDownBlock(in_channels, f,     normalize=False)  # 3→64
        self.enc2 = UNetDownBlock(f,          f * 2)                     # 64→128
        self.enc3 = UNetDownBlock(f * 2,      f * 4)                     # 128→256
        self.enc4 = UNetDownBlock(f * 4,      f * 8)                     # 256→512

        # ── 瓶颈层 ────────────────────────────────────────────────────
        self.bottleneck = UNetDownBlock(f * 8, f * 8)                    # 512→512

        # ── 解码器 (4 层上采样 + skip connection) ─────────────────────
        #   ConvTranspose 输出通道数 ≤ 编码器对应层通道数, 拼接 skip 后翻倍
        self.dec4 = UNetUpBlock(f * 8,  f * 8, dropout=0.5)             # 512→512, +skip 512 → 1024
        self.dec3 = UNetUpBlock(f * 16, f * 4, dropout=0.5)             # 1024→256, +skip 256 → 512
        self.dec2 = UNetUpBlock(f * 8,  f * 2)                          # 512→128, +skip 128 → 256
        self.dec1 = UNetUpBlock(f * 4,  f)                              # 256→64,  +skip 64  → 128

        # ── 输出头 ────────────────────────────────────────────────────
        self.output = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # 权重初始化
        self._init_weights()

    # ── 权重初始化 ────────────────────────────────────────────────────
    def _init_weights(self):
        """Conv2d / ConvTranspose2d: Normal(0, 0.02); BN: weight=1, bias=0"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── 前向传播 ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB 输入 [B, 3, H, W]   (H, W 需为 32 的倍数)

        Returns:
            NIR 输出 [B, 1, H, W], 值域 [-1, 1]
        """
        # 编码器 — 保存每层特征用于 skip connection
        e1 = self.enc1(x)         # [B,  64, H/2,  W/2]
        e2 = self.enc2(e1)        # [B, 128, H/4,  W/4]
        e3 = self.enc3(e2)        # [B, 256, H/8,  W/8]
        e4 = self.enc4(e3)        # [B, 512, H/16, W/16]

        # 瓶颈层
        b = self.bottleneck(e4)   # [B, 512, H/32, W/32]

        # 解码器 — 拼接 skip connection
        d4 = self.dec4(b,  e4)    # [B, 1024, H/16, W/16]
        d3 = self.dec3(d4, e3)    # [B,  512, H/8,  W/8]
        d2 = self.dec2(d3, e2)    # [B,  256, H/4,  W/4]
        d1 = self.dec1(d2, e1)    # [B,  128, H/2,  W/2]

        # 输出头
        return self.output(d1)    # [B,    1, H,    W]


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = UNetGenerator(in_channels=3, out_channels=1).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}\n")

    # 验证不同尺寸
    for size in [256, 512]:
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"[{size}×{size}]  Input: {list(x.shape)}  →  Output: {list(y.shape)}")
        assert y.shape == (1, 1, size, size), \
            f"Shape mismatch! Expected [1,1,{size},{size}], got {list(y.shape)}"
        assert y.min() >= -1.0 and y.max() <= 1.0, \
            "Output values out of [-1, 1] range!"
        print(f"            Range: [{y.min().item():.4f}, {y.max().item():.4f}]  ✓\n")

    print("✅ All tests passed.")
