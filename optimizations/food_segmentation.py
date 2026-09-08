"""
优化5: 食物分割前置
====================
在多任务网络之前先做食物区域分割，去除背景干扰，
只保留食物区域特征送入下游网络。

思路:
    1. 使用轻量级分割网络（如U-Net/FastSCNN）对输入图像做语义分割
    2. 分割结果生成食物区域mask
    3. 原图 × mask → 去背景图 → 送入多任务网络
    4. 分割模块可预训练后冻结，也可端到端微调

    优势:
    - 去除盘子、桌面、手等背景干扰
    - 让NIR生成器和多任务网络只关注食物区域
    - 分割mask可复用给空间注意力模块

实现:
    分割网络基于U-Net架构，输出2类: 食物 / 背景
    支持加载预训练权重或从头训练
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class SegUNetDown(nn.Module):
    """分割U-Net下采样块"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegUNetUp(nn.Module):
    """分割U-Net上采样块"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 处理尺寸不匹配
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FoodSegmentationNet(nn.Module):
    """食物区域分割网络

    轻量级U-Net，输出食物区域二值mask。

    Args:
        input_channels: 输入通道数 (3=RGB 或 4=RGB+NIR)
        num_classes: 分割类别数 (默认2: 食物/背景)
        base_features: 基础特征数
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        base_features: int = 32,
    ):
        super().__init__()
        f = base_features

        # 编码器
        self.enc1 = SegUNetDown(input_channels, f)       # → [f, H/2, W/2]
        self.enc2 = SegUNetDown(f, f * 2)                 # → [2f, H/4, W/4]
        self.enc3 = SegUNetDown(f * 2, f * 4)             # → [4f, H/8, W/8]
        self.enc4 = SegUNetDown(f * 4, f * 8)             # → [8f, H/16, W/16]

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 16, 3, 1, 1),
            nn.BatchNorm2d(f * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 16, f * 16, 3, 1, 1),
            nn.BatchNorm2d(f * 16),
            nn.ReLU(inplace=True),
        )

        # 解码器
        self.dec4 = SegUNetUp(f * 16, f * 8)
        self.dec3 = SegUNetUp(f * 8, f * 4)
        self.dec2 = SegUNetUp(f * 4, f * 2)
        self.dec1 = SegUNetUp(f * 2, f)

        # 输出层
        self.final = nn.Conv2d(f, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            dict: {
                'logits': [B, 2, H, W]     分割logits
                'mask':   [B, 1, H, W]     食物区域概率mask (0~1)
            }
        """
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 瓶颈
        b = self.bottleneck(e4)

        # 解码器 + skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # 输出
        logits = self.final(d1)  # [B, 2, H, W]

        # 食物区域概率（类别1 = 食物）
        mask = torch.softmax(logits, dim=1)[:, 1:2, :, :]  # [B, 1, H, W]

        return {
            "logits": logits,
            "mask": mask,
        }


class FoodSegmentationPreprocessor(nn.Module):
    """食物分割预处理器

    将食物分割网络封装为预处理模块:
    1. 对输入RGB图做分割
    2. 用mask去除背景
    3. 输出去背景后的图像和mask

    Args:
        seg_model: 分割网络（或路径）
        freeze_segmentation: 是否冻结分割网络参数
        mask_threshold: mask二值化阈值 (默认0.5)
    """

    def __init__(
        self,
        seg_model: Optional[FoodSegmentationNet] = None,
        freeze_segmentation: bool = True,
        mask_threshold: float = 0.5,
    ):
        super().__init__()
        if seg_model is None:
            seg_model = FoodSegmentationNet(input_channels=3, num_classes=2)

        self.seg_model = seg_model
        self.mask_threshold = mask_threshold

        if freeze_segmentation:
            for param in self.seg_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        image: torch.Tensor,
        soft_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            image: 输入RGB图像 [B, 3, H, W]
            soft_mask: 使用软mask(True)或硬mask(False)

        Returns:
            dict: {
                'masked_image': [B, 3, H, W]  去背景图像
                'mask':         [B, 1, H, W]  食物区域mask
                'seg_logits':   [B, 2, H, W]  分割logits
            }
        """
        # 分割
        seg_out = self.seg_model(image)
        mask = seg_out["mask"]

        if not soft_mask:
            mask = (mask > self.mask_threshold).float()

        # 去背景
        masked_image = image * mask

        return {
            "masked_image": masked_image,
            "mask": mask,
            "seg_logits": seg_out["logits"],
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试食物分割网络
    seg_net = FoodSegmentationNet(input_channels=3, num_classes=2).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = seg_net(x)
    print(f"分割logits: {out['logits'].shape}")
    print(f"食物mask: {out['mask'].shape}, 范围 [{out['mask'].min():.3f}, {out['mask'].max():.3f}]")
    print(f"参数量: {sum(p.numel() for p in seg_net.parameters()):,}")

    # 测试预处理器
    preprocessor = FoodSegmentationPreprocessor(freeze_segmentation=True).to(device)
    with torch.no_grad():
        result = preprocessor(x)
    print(f"去背景图像: {result['masked_image'].shape}")
