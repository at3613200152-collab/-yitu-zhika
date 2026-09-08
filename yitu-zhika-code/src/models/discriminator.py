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
    ------
    L1(k4,s2): RF=4
    L2(k4,s2): RF=4+3×2 = 10
    L3(k4,s2): RF=10+3×4 = 22
    L4(k4,s1): RF=22+3×8 = 46
    Out(k4,s1): RF=46+3×8 = 70  ✓

权重初始化: Conv Normal(0, 0.02), BN weight=1 bias=0
参考: Isola et al., CVPR 2017
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN 条件判别器。

    输入 RGB 条件图和 NIR 目标图（真或假），输出每个 70×70
    局部区域的真/假 logits。

    Args:
        in_channels: 输入通道数 (RGB 3 + NIR 1 = 4)
        ndf: 首层卷积通道数 (默认 64, 逐层翻倍)
    """

    def __init__(self, in_channels: int = 4, ndf: int = 64):
        super().__init__()

        kw = 4   # kernel size
        pad = 1  # padding

        self.model = nn.Sequential(
            # ── Layer 1: Conv → BN → LeakyReLU (stride 2) ────────────
            nn.Conv2d(in_channels, ndf, kw, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            # ── Layer 2: Conv → BN → LeakyReLU (stride 2) ────────────
            nn.Conv2d(ndf, ndf * 2, kw, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # ── Layer 3: Conv → BN → LeakyReLU (stride 2) ────────────
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # ── Layer 4: Conv → BN → LeakyReLU (stride 1) ────────────
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ── Output: Conv → 1 通道 logits (stride 1) ─────────────
            nn.Conv2d(ndf * 8, 1, kw, stride=1, padding=pad, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        """Conv: Normal(0, 0.02); BatchNorm: weight=1, bias=0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            rgb: RGB 条件图 [B, 3, H, W]
            nir: NIR 目标图（真或假）[B, 1, H, W]

        Returns:
            logits 图 [B, 1, H', W']（未过 Sigmoid，配合 BCEWithLogitsLoss）
        """
        x = torch.cat([rgb, nir], dim=1)  # [B, 4, H, W]
        return self.model(x)

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        """推理接口：返回 [0, 1] 概率图。"""
        return torch.sigmoid(self.forward(rgb, nir))


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = PatchDiscriminator(in_channels=4, ndf=64).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}\n")

    for size in [256, 512]:
        rgb = torch.randn(2, 3, size, size, device=device)
        nir = torch.randn(2, 1, size, size, device=device)
        with torch.no_grad():
            out = model(rgb, nir)
        print(f"[{size}×{size}]  Input: rgb {list(rgb.shape)} + nir {list(nir.shape)}"
              f"  →  Output: {list(out.shape)}")
        assert out.shape[0] == 2 and out.shape[1] == 1
        print(f"            Range: [{out.min().item():.4f}, {out.max().item():.4f}]  ✓\n")

    # 测试 predict
    prob = model.predict(rgb, nir)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    print(f"Predict range: [{prob.min().item():.4f}, {prob.max().item():.4f}]  ✓")

    print("\n✅ All tests passed.")
