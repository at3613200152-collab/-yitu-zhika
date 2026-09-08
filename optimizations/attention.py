"""
优化3: 空间注意力聚焦食物区域
==============================
基线模型对整张图均匀处理，但食物通常只占图像部分区域。
空间注意力模块引导模型聚焦食物区域，抑制背景干扰。

思路:
    1. 空间注意力图: 对特征图沿通道维度计算注意力权重
    2. 通道注意力: 对特征图沿空间维度计算通道重要性
    3. CBAM (Convolutional Block Attention Module) 风格:
       通道注意力 → 空间注意力 → 逐元素调制

    在ResNet多任务网络的layer3/layer4后插入注意力模块，
    让模型自动聚焦食物区域，提升分类和回归精度。

参考: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
"""

import torch
import torch.nn as nn
from typing import Optional


class ChannelAttention(nn.Module):
    """通道注意力模块

    通过全局平均池化和最大池化捕获通道间依赖关系。

    Args:
        in_channels: 输入通道数
        reduction: 降维比例 (默认16)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            通道注意力权重 [B, C, 1, 1]
        """
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    """空间注意力模块

    沿通道维度计算平均和最大值，生成空间注意力图，
    聚焦食物区域，抑制背景。

    Args:
        kernel_size: 卷积核大小 (默认7)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            空间注意力图 [B, 1, H, W]
        """
        avg_out = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attn = torch.sigmoid(self.conv(cat))  # [B, 1, H, W]
        return attn


class CBAM(nn.Module):
    """CBAM: 通道注意力 + 空间注意力

    顺序应用通道注意力和空间注意力对特征图进行调制。

    Args:
        in_channels: 输入通道数
        reduction: 通道注意力降维比例
        spatial_kernel: 空间注意力卷积核大小
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            调制后的特征 [B, C, H, W]
        """
        # 通道注意力
        x = x * self.channel_attn(x)
        # 空间注意力
        x = x * self.spatial_attn(x)
        return x


class FoodRegionAttention(nn.Module):
    """食物区域注意力模块

    专门针对食物图像设计的注意力模块，包含:
    1. CBAM基础注意力
    2. 食物区域掩码引导（如果提供食物分割mask）

    当有食物分割mask时，直接用mask加权特征图；
    没有mask时，依靠CBAM自动学习注意力。

    Args:
        in_channels: 输入通道数
        reduction: 通道注意力降维比例
        use_mask_guidance: 是否使用食物分割mask引导
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        use_mask_guidance: bool = True,
    ):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction)
        self.use_mask_guidance = use_mask_guidance

        if use_mask_guidance:
            # mask引导: 1x1卷积将mask映射为注意力权重
            self.mask_guidance = nn.Sequential(
                nn.Conv2d(1, in_channels // 4, 1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1, bias=False),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        food_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]
            food_mask: 食物区域掩码 [B, 1, H, W] (可选)

        Returns:
            调制后的特征 [B, C, H, W]
        """
        # CBAM自注意力
        x = self.cbam(x)

        # Mask引导（如果提供）
        if self.use_mask_guidance and food_mask is not None:
            mask_attn = self.mask_guidance(food_mask)  # [B, 1, H, W]
            x = x * mask_attn

        return x


def insert_attention_to_resnet(
    model: nn.Module,
    attention_positions: list = ['layer3', 'layer4'],
    reduction: int = 16,
) -> nn.Module:
    """将注意力模块插入ResNet的指定层

    在ResNet的layer3/layer4后插入CBAM模块。

    Args:
        model: ResNet模型
        attention_positions: 插入注意力的层名列表
        reduction: 通道注意力降维比例

    Returns:
        修改后的模型
    """
    for layer_name in attention_positions:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            # 获取该层的输出通道数
            # ResNet layer3输出512, layer4输出2048 (对ResNet50)
            out_channels = layer[-1].bn3.num_features if hasattr(layer[-1], 'bn3') else 512
            attn = CBAM(out_channels, reduction)

            # 将注意力模块附加到层末尾
            # 使用nn.Sequential包装
            original_forward = layer.forward

            def make_attn_forward(orig_fwd, attn_module):
                def new_forward(x):
                    return attn_module(orig_fwd(x))
                return new_forward

            layer.forward = make_attn_forward(original_forward, attn)
            # 将attn模块注册为子模块以确保参数被追踪
            model.add_module(f'{layer_name}_attn', attn)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试CBAM
    x = torch.randn(2, 512, 32, 32).to(device)
    cbam = CBAM(512, reduction=16).to(device)
    y = cbam(x)
    print(f"CBAM: 输入 {x.shape} → 输出 {y.shape}")

    # 测试食物区域注意力
    fra = FoodRegionAttention(512, use_mask_guidance=True).to(device)
    mask = torch.ones(2, 1, 32, 32).to(device)
    y = fra(x, mask)
    print(f"FoodRegionAttention: 输入 {x.shape} + mask → 输出 {y.shape}")
