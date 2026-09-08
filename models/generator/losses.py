"""
生成器损失函数
==============
包含GAN损失、L1重建损失和感知损失(VGG Perceptual Loss)。

总生成器损失:
    G_loss = λ_GAN * GAN_loss + λ_L1 * L1_loss + λ_perceptual * Perceptual_loss

感知损失使用预训练VGG19提取多层特征，比较生成图和真实图的特征差异，
相比纯L1损失，感知损失能更好地保持图像的结构和纹理信息。
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional, Tuple


class VGGPerceptualLoss(nn.Module):
    """VGG感知损失

    使用预训练VGG19网络提取多层特征，计算特征空间中的L1距离。
    
    默认使用 conv3_4, conv4_4, conv5_4 三层特征，权重分别为 1/32, 1/16, 1/8。

    Args:
        feature_layers: 使用VGG的哪些层特征 (默认 [8, 17, 26])
        feature_weights: 各层特征权重 (默认 [1/32, 1/16, 1/8])
        normalize_input: 是否将输入从[-1,1]归一化到VGG期望的格式
    """

    # VGG19各层对应索引:
    # conv1_2=1, conv2_2=6, conv3_2=11, conv3_4=17, conv4_4=26, conv5_4=35
    LAYER_INDICES = {
        'conv1_2': 1, 'conv2_2': 6, 'conv3_2': 11,
        'conv3_4': 17, 'conv4_4': 26, 'conv5_4': 35,
    }

    def __init__(
        self,
        feature_layers: Optional[List[int]] = None,
        feature_weights: Optional[List[float]] = None,
        normalize_input: bool = True,
    ):
        super().__init__()
        # 默认使用 conv3_4, conv4_4, conv5_4
        if feature_layers is None:
            feature_layers = [17, 26, 35]
        if feature_weights is None:
            feature_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8]

        self.feature_layers = feature_layers
        self.feature_weights = feature_weights
        self.normalize_input = normalize_input

        # 加载预训练VGG19特征提取器（只保留特征层，去掉分类头）
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()

        # 将VGG按feature_layers切分成多个子模块
        prev_idx = 0
        for layer_idx in sorted(feature_layers):
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:layer_idx + 1]))
            prev_idx = layer_idx + 1

        # 冻结VGG参数
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算感知损失

        Args:
            pred: 预测图像 [B, C, H, W], 值域[-1, 1]
            target: 目标图像 [B, C, H, W], 值域[-1, 1]

        Returns:
            感知损失标量
        """
        # 如果输入是单通道(NIR)，复制为3通道以适配VGG
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 归一化: [-1, 1] → ImageNet标准格式
        if self.normalize_input:
            pred = (pred + 1) / 2  # → [0, 1]
            target = (target + 1) / 2
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0.0
        x_pred = pred
        x_target = target

        for i, slice_model in enumerate(self.slices):
            x_pred = slice_model(x_pred)
            with torch.no_grad():
                x_target = slice_model(x_target)
            loss += self.feature_weights[i] * nn.functional.l1_loss(x_pred, x_target)

        return loss


class GeneratorLoss(nn.Module):
    """生成器综合损失

    G_loss = λ_GAN * GAN_loss + λ_L1 * L1_loss + λ_perceptual * Perceptual_loss

    Args:
        lambda_gan: GAN损失权重
        lambda_l1: L1重建损失权重
        lambda_perceptual: 感知损失权重
        use_perceptual: 是否启用感知损失
    """

    def __init__(
        self,
        lambda_gan: float = 1.0,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss() if use_perceptual else None

    def forward(
        self,
        pred_nir: torch.Tensor,
        real_nir: torch.Tensor,
        disc_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """计算生成器综合损失

        Args:
            pred_nir: 生成器预测的NIR图像 [B, 1, H, W]
            real_nir: 真实NIR图像 [B, 1, H, W]
            disc_pred: 判别器对假图对的输出 [B, 1, H', W']

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失明细
        """
        # GAN损失: 生成器希望判别器认为假图是真图
        valid = torch.ones_like(disc_pred)
        loss_gan = self.criterion_gan(disc_pred, valid)

        # L1重建损失
        loss_l1 = self.criterion_l1(pred_nir, real_nir)

        # 感知损失
        loss_perceptual = torch.tensor(0.0, device=pred_nir.device)
        if self.use_perceptual and self.criterion_perceptual is not None:
            loss_perceptual = self.criterion_perceptual(pred_nir, real_nir)

        # 总损失
        total_loss = (
            self.lambda_gan * loss_gan
            + self.lambda_l1 * loss_l1
            + self.lambda_perceptual * loss_perceptual
        )

        loss_dict = {
            "g_total": total_loss.item(),
            "g_gan": (self.lambda_gan * loss_gan).item(),
            "g_l1": (self.lambda_l1 * loss_l1).item(),
            "g_perceptual": (self.lambda_perceptual * loss_perceptual).item(),
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试感知损失
    perceptual_loss = VGGPerceptualLoss().to(device)
    pred = torch.randn(2, 1, 256, 256).to(device)
    target = torch.randn(2, 1, 256, 256).to(device)
    loss = perceptual_loss(pred, target)
    print(f"感知损失: {loss.item():.4f}")

    # 测试综合生成器损失
    gen_loss = GeneratorLoss(
        lambda_gan=1.0, lambda_l1=100.0, lambda_perceptual=10.0
    ).to(device)
    disc_out = torch.randn(2, 1, 30, 30).to(device)
    total_loss, loss_dict = gen_loss(pred, target, disc_out)
    print(f"生成器综合损失: {loss_dict}")
