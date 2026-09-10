"""
改造ResNet多任务网络 (v2 — 5回归头)
=====================================
基于ResNet50骨干网络，同时进行食物分类和营养回归。

v2改动:
    回归头从2维扩展到5维: [卡路里, 重量, 蛋白质, 碳水, 脂肪]

架构:
    Input [B, 4, H, W]
      → ResNet50 Backbone (conv1修改为4通道)
      → AdaptiveAvgPool2d → [B, 2048]
      → 分类头: FC → num_classes
      → 回归头: FC(2048→512→5) + ReLU
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any, Tuple


# 回归目标维度: [卡路里(kcal), 重量(g), 蛋白质(g), 碳水(g), 脂肪(g)]
NUM_NUTRITION_TARGETS = 5

# 各回归目标的均值/标准差，用于标准化（可选，不标准化则直接回归原始值）
# 基于Nutrition5k统计
NUTRITION_STATS = {
    "calories": {"mean": 355.0, "std": 180.0},
    "mass":     {"mean": 350.0, "std": 150.0},
    "protein":  {"mean":  20.0, "std":  12.0},
    "carb":     {"mean":  35.0, "std":  20.0},
    "fat":      {"mean":  15.0, "std":  10.0},
}


class ResNetMultiTask(nn.Module):
    """ResNet50 多任务网络 (v2)

    同时输出食物分类和营养回归(卡路里+重量+蛋白质+碳水+脂肪)。

    Args:
        num_classes: 食物类别数
        input_channels: 输入通道数 (默认4, RGB+NIR)
        pretrained: 是否加载ImageNet预训练权重
        dropout_rate: Dropout概率
        hidden_dim: 回归头隐藏层维度
        num_regression_targets: 回归输出维度 (默认5)
        normalize_targets: 是否对回归目标做标准化
    """

    def __init__(
        self,
        num_classes: int = 61,
        input_channels: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
        num_regression_targets: int = NUM_NUTRITION_TARGETS,
        normalize_targets: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.num_regression_targets = num_regression_targets
        self.normalize_targets = normalize_targets

        # 加载ResNet50骨干
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet50(weights=None)

        # ========= 修改第一层卷积 =========
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight[:, :3, :, :].clone()
                if input_channels > 3:
                    # NIR 通道用 RGB 均值初始化 × 0.3（原 0.01 太小，NIR 通道几乎不贡献）
                    new_conv1.weight[:, 3:, :, :] = original_conv1.weight[:, :1, :, :].mean(
                        dim=1, keepdim=True
                    ).expand(-1, input_channels - 3, -1, -1) * 0.3

        backbone.conv1 = new_conv1

        # ========= 提取特征层 =========
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.avgpool = backbone.avgpool
        self.feature_dim = backbone.fc.in_features  # 2048

        # ========= 分类头 =========
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

        # ========= 回归头 (v2: 5维输出) =========
        # 2048 → 512 → num_regression_targets
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_regression_targets),
        )

        # 标准化参数（可选）
        if normalize_targets:
            stats_vals = torch.tensor([
                NUTRITION_STATS["calories"]["mean"],
                NUTRITION_STATS["mass"]["mean"],
                NUTRITION_STATS["protein"]["mean"],
                NUTRITION_STATS["carb"]["mean"],
                NUTRITION_STATS["fat"]["mean"],
            ])
            stats_stds = torch.tensor([
                NUTRITION_STATS["calories"]["std"],
                NUTRITION_STATS["mass"]["std"],
                NUTRITION_STATS["protein"]["std"],
                NUTRITION_STATS["carb"]["std"],
                NUTRITION_STATS["fat"]["std"],
            ])
            self.register_buffer("target_mean", stats_vals)
            self.register_buffer("target_std", stats_stds)
        else:
            self.target_mean = None
            self.target_std = None

        self._init_weights()

    def _init_weights(self):
        for module in [self.classifier, self.regressor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            x: 输入图像 [B, 4, H, W] (RGB+NIR拼接)

        Returns:
            dict: {
                'logits':    [B, num_classes]      分类logits
                'nutrition':  [B, 5]               回归输出 [卡路里, 重量, 蛋白质, 碳水, 脂肪]
                'features':   [B, 2048]            共享特征
            }
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        logits = self.classifier(feat)
        nutrition = self.regressor(feat)

        return {
            "logits": logits,
            "nutrition": nutrition,
            "features": feat,
        }

    def get_classification_output(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)

    def get_regression_output(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.regressor(feat)

    def normalize_regression_output(self, nutrition: torch.Tensor) -> torch.Tensor:
        """如果训练时用了标准化，推理时反标准化到原始值域"""
        if self.normalize_targets and self.target_mean is not None:
            return nutrition * self.target_std + self.target_mean
        return nutrition


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = ResNetMultiTask(
        num_classes=61,
        input_channels=4,
        pretrained=True,
        num_regression_targets=5,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    x = torch.randn(4, 4, 256, 256).to(device)
    with torch.no_grad():
        outputs = model(x)

    print(f"分类输出: {outputs['logits'].shape}")    # [4, 61]
    print(f"回归输出: {outputs['nutrition'].shape}")  # [4, 5]
    print(f"特征输出: {outputs['features'].shape}")    # [4, 2048]
    print(f"\n回归维度: {model.num_regression_targets} (卡路里, 重量, 蛋白质, 碳水, 脂肪)")
