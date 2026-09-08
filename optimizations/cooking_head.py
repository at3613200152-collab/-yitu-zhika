"""
优化4: 烹饪方式辅助分类头
=========================
不同烹饪方式对同一食材的卡路里影响极大（如水煮鸡胸 vs 油炸鸡胸）。
本模块添加一个烹饪方式辅助分类头，与主分类头联合训练。

思路:
    1. 烹饪方式分类: 蒸/煮/炒/炸/烤/生食/凉拌 等
    2. 作为辅助任务与主分类和回归任务联合训练
    3. 烹饪方式特征参与卡路里回归，提升精度
    4. 推理时可选择是否使用烹饪方式输出

    烹饪方式对卡路里的影响示例:
    - 水煮鸡胸 165 kcal/100g
    - 红烧鸡胸 200 kcal/100g  
    - 油炸鸡胸 280 kcal/100g
    同一食材不同做法差异可达70%

实现:
    在多任务ResNet的共享特征后，添加烹饪方式分类头，
    并将烹饪方式的嵌入向量拼接到回归头输入中。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


# 常见烹饪方式
COOKING_METHODS = [
    "生食",      # raw
    "蒸",        # steamed
    "煮",        # boiled
    "炒",        # stir-fried
    "炸",        # deep-fried
    "烤",        # roasted/baked
    "炖",        # braised/stewed
    "凉拌",      # cold dish
    "煎",        # pan-fried
    "微波",      # microwaved
]

# 烹饪方式对卡路里的典型影响系数（相对于生食）
COOKING_CALORIE_FACTORS = {
    "生食": 1.0,
    "蒸": 1.05,
    "煮": 1.08,
    "炒": 1.35,
    "炸": 1.70,
    "烤": 1.15,
    "炖": 1.20,
    "凉拌": 1.10,
    "煎": 1.45,
    "微波": 1.05,
}


class CookingMethodHead(nn.Module):
    """烹饪方式分类头

    基于共享特征预测食物的烹饪方式。

    Args:
        feature_dim: 输入特征维度 (ResNet50 = 2048)
        num_cooking_methods: 烹饪方式类别数
        hidden_dim: 隐藏层维度
        dropout_rate: Dropout概率
        embed_dim: 烹饪方式嵌入维度（用于回归头）
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_cooking_methods: int = 10,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.num_cooking_methods = num_cooking_methods
        self.embed_dim = embed_dim

        # 分类头: 特征 → 烹饪方式类别
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_cooking_methods),
        )

        # 嵌入层: 烹饪方式类别 → 语义嵌入
        self.embedding = nn.Embedding(num_cooking_methods, embed_dim)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            features: 共享特征 [B, feature_dim]

        Returns:
            dict: {
                'logits':     [B, num_cooking_methods]  分类logits
                'embeddings': [B, embed_dim]            烹饪方式嵌入
                'predicted':  [B]                       预测类别索引
            }
        """
        logits = self.classifier(features)
        predicted = logits.argmax(dim=-1)
        embeddings = self.embedding(predicted)

        return {
            "logits": logits,
            "embeddings": embeddings,
            "predicted": predicted,
        }


class CalorieRegressorWithCooking(nn.Module):
    """融合烹饪方式的卡路里回归头

    将食物类别特征和烹饪方式嵌入拼接后回归卡路里和重量。

    Args:
        feature_dim: 食物特征维度
        cooking_embed_dim: 烹饪方式嵌入维度
        hidden_dim: 隐藏层维度
        dropout_rate: Dropout概率
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        cooking_embed_dim: int = 32,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        input_dim = feature_dim + cooking_embed_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # [卡路里, 重量]
        )

    def forward(
        self,
        features: torch.Tensor,
        cooking_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播

        Args:
            features: 食物特征 [B, feature_dim]
            cooking_embeddings: 烹饪方式嵌入 [B, cooking_embed_dim]

        Returns:
            营养预测 [B, 2] (卡路里, 重量)
        """
        x = torch.cat([features, cooking_embeddings], dim=1)
        return self.regressor(x)


class MultiTaskWithCooking(nn.Module):
    """带烹饪方式的多任务网络

    在基线多任务网络基础上增加烹饪方式辅助分类头。

    Args:
        feature_dim: 共享特征维度 (ResNet50 = 2048)
        num_food_classes: 食物类别数
        num_cooking_methods: 烹饪方式类别数
        cooking_embed_dim: 烹饪方式嵌入维度
        dropout_rate: Dropout概率
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_food_classes: int = 61,
        num_cooking_methods: int = 10,
        cooking_embed_dim: int = 32,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # 食物分类头
        self.food_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_food_classes),
        )

        # 烹饪方式头
        self.cooking_head = CookingMethodHead(
            feature_dim=feature_dim,
            num_cooking_methods=num_cooking_methods,
            embed_dim=cooking_embed_dim,
        )

        # 融合烹饪方式的回归头
        self.nutrition_regressor = CalorieRegressorWithCooking(
            feature_dim=feature_dim,
            cooking_embed_dim=cooking_embed_dim,
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            features: ResNet共享特征 [B, feature_dim]

        Returns:
            dict: {
                'food_logits':   [B, num_food_classes]    食物分类logits
                'cooking_logits': [B, num_cooking_methods] 烹饪方式logits
                'nutrition':     [B, 2]                   营养回归
                'cooking_pred':  [B]                      烹饪方式预测
            }
        """
        # 食物分类
        food_logits = self.food_classifier(features)

        # 烹饪方式
        cooking_out = self.cooking_head(features)

        # 营养回归（融合烹饪方式嵌入）
        nutrition = self.nutrition_regressor(features, cooking_out["embeddings"])

        return {
            "food_logits": food_logits,
            "cooking_logits": cooking_out["logits"],
            "nutrition": nutrition,
            "cooking_pred": cooking_out["predicted"],
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试烹饪方式头
    features = torch.randn(4, 2048).to(device)
    cooking_head = CookingMethodHead(feature_dim=2048, num_cooking_methods=10).to(device)
    out = cooking_head(features)
    print(f"烹饪方式logits: {out['logits'].shape}")
    print(f"烹饪方式嵌入: {out['embeddings'].shape}")

    # 测试完整多任务网络
    model = MultiTaskWithCooking(
        feature_dim=2048, num_food_classes=61, num_cooking_methods=10,
    ).to(device)
    outputs = model(features)
    print(f"食物分类: {outputs['food_logits'].shape}")
    print(f"烹饪方式: {outputs['cooking_logits'].shape}")
    print(f"营养回归: {outputs['nutrition'].shape}")

    # 打印烹饪方式卡路里影响系数
    print("\n烹饪方式卡路里影响系数:")
    for method, factor in COOKING_CALORIE_FACTORS.items():
        print(f"  {method}: ×{factor:.2f}")
