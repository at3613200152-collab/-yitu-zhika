"""
改造ResNet多任务网络
====================
基于ResNet50骨干网络，同时进行食物分类和营养回归。

关键改造:
    1. 第一层conv从3通道改为4通道 (RGB+NIR)
    2. 预训练权重处理: 前3通道拷贝ImageNet权重，第4通道初始化为0
    3. 替换最后的全连接层为两个任务头:
       - 分类头: → num_classes (CrossEntropy)
       - 回归头: → 2 (卡路里, 重量) (L1 + MAPE)

架构:
    Input [B, 4, H, W]
      → ResNet50 Backbone (conv1修改为4通道)
      → AdaptiveAvgPool2d → [B, 2048]
      → 分类头: FC → num_classes
      → 回归头: FC(2048→512→2) + ReLU
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any, Tuple


class ResNetMultiTask(nn.Module):
    """ResNet50 多任务网络

    同时输出食物分类和营养回归(卡路里+重量)。

    Args:
        num_classes: 食物类别数 (默认61)
        input_channels: 输入通道数 (默认4, RGB+NIR)
        pretrained: 是否加载ImageNet预训练权重
        dropout_rate: Dropout概率 (默认0.5)
        hidden_dim: 回归头隐藏层维度 (默认512)
    """

    def __init__(
        self,
        num_classes: int = 61,
        input_channels: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # 加载ResNet50骨干
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet50(weights=None)

        # ========= 修改第一层卷积 =========
        # 原始: Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # 前3通道拷贝预训练权重，第4通道(NIR)初始化为0
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight[:, :3, :, :].clone()
                if input_channels > 3:
                    # NIR通道用均值初始化（比全零更好收敛）
                    new_conv1.weight[:, 3:, :, :] = original_conv1.weight[:, :1, :, :].mean(
                        dim=1, keepdim=True
                    ).expand(-1, input_channels - 3, -1, -1) * 0.01

        backbone.conv1 = new_conv1

        # ========= 提取特征层（去掉最后FC） =========
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

        # 全局平均池化
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d(1)

        # 特征维度
        self.feature_dim = backbone.fc.in_features  # 2048

        # ========= 分类头 =========
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

        # ========= 回归头 =========
        # 两层MLP: 2048 → 512 → 2 (卡路里, 重量)
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 2),  # [卡路里, 重量]
        )

        # 初始化新增层的权重
        self._init_weights()

    def _init_weights(self):
        """初始化分类头和回归头的权重"""
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
                'logits':   [B, num_classes]  分类logits
                'nutrition': [B, 2]           回归输出 [卡路里, 重量]
                'features': [B, 2048]         共享特征（可用于可视化）
            }
        """
        # 提取特征
        feat = self.features(x)          # [B, 2048, H', W']
        feat = self.avgpool(feat)        # [B, 2048, 1, 1]
        feat = torch.flatten(feat, 1)    # [B, 2048]

        # 分类头
        logits = self.classifier(feat)   # [B, num_classes]

        # 回归头
        nutrition = self.regressor(feat) # [B, 2]

        return {
            "logits": logits,
            "nutrition": nutrition,
            "features": feat,
        }

    def get_classification_output(self, x: torch.Tensor) -> torch.Tensor:
        """只获取分类输出（推理加速）"""
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)

    def get_regression_output(self, x: torch.Tensor) -> torch.Tensor:
        """只获取回归输出"""
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.regressor(feat)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = ResNetMultiTask(
        num_classes=61,
        input_channels=4,
        pretrained=True,
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(4, 4, 256, 256).to(device)
    with torch.no_grad():
        outputs = model(x)

    print(f"分类输出: {outputs['logits'].shape}")      # [4, 61]
    print(f"回归输出: {outputs['nutrition'].shape}")    # [4, 2]
    print(f"特征输出: {outputs['features'].shape}")     # [4, 2048]
