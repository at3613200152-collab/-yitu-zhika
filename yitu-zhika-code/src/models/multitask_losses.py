"""
多任务损失函数 v2 (5回归头)
=============================
包含分类损失 + 5个回归目标的 L1+MAPE 损失。

总损失:
    L = λ_cls * CrossEntropy(logits, labels)
      + Σ_i [λ_i * L1(pred_i, gt_i)]         (i ∈ {cal, weight, protein, carb, fat})
      + λ_mape * Σ_i [MAPE(pred_i, gt_i)] / 5
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


# 回归目标名称（顺序与模型输出一致）
REGRESSION_NAMES = ["calories", "weight", "protein", "carb", "fat"]


class MAPELoss(nn.Module):
    """MAPE (Mean Absolute Percentage Error) 损失"""

    def __init__(self, epsilon: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = torch.abs(target) + self.epsilon
        mape = torch.abs(pred - target) / denom
        if self.reduction == "mean":
            return mape.mean()
        elif self.reduction == "sum":
            return mape.sum()
        return mape


class MultiTaskLossV2(nn.Module):
    """多任务综合损失 v2

    支持5个回归目标: [卡路里, 重量, 蛋白质, 碳水, 脂肪]

    Args:
        num_classes: 分类类别数
        lambda_cls: 分类损失权重
        lambda_cal: 卡路里L1权重
        lambda_weight: 重量L1权重
        lambda_protein: 蛋白质L1权重
        lambda_carb: 碳水L1权重
        lambda_fat: 脂肪L1权重
        lambda_mape: MAPE损失权重（所有目标平均）
        label_smoothing: 标签平滑
        mape_epsilon: MAPE防除零
    """

    def __init__(
        self,
        num_classes: int = 61,
        lambda_cls: float = 1.0,
        lambda_cal: float = 1.0,
        lambda_weight: float = 0.5,
        lambda_protein: float = 0.3,
        lambda_carb: float = 0.3,
        lambda_fat: float = 0.3,
        lambda_mape: float = 0.1,
        label_smoothing: float = 0.1,
        mape_epsilon: float = 1.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_cal = lambda_cal
        self.lambda_weight = lambda_weight
        self.lambda_protein = lambda_protein
        self.lambda_carb = lambda_carb
        self.lambda_fat = lambda_fat
        self.lambda_mape = lambda_mape

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mape = MAPELoss(epsilon=mape_epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        nutrition_pred: torch.Tensor,
        nutrition_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算多任务综合损失

        Args:
            logits: [B, num_classes]
            labels: [B]
            nutrition_pred: [B, 5] 预测 [卡路里, 重量, 蛋白质, 碳水, 脂肪]
            nutrition_gt:   [B, 5] 真值

        Returns:
            total_loss, loss_dict
        """
        # 分类损失
        loss_cls = self.criterion_cls(logits, labels)

        # 各回归目标 L1
        lambdas = [
            self.lambda_cal, self.lambda_weight,
            self.lambda_protein, self.lambda_carb, self.lambda_fat,
        ]
        l1_losses = []
        mape_losses = []

        for i, name in enumerate(REGRESSION_NAMES):
            pred_i = nutrition_pred[:, i]
            gt_i = nutrition_gt[:, i]
            l1_i = self.criterion_l1(pred_i, gt_i)
            mape_i = self.criterion_mape(pred_i, gt_i)
            l1_losses.append(l1_i)
            mape_losses.append(mape_i)

        # 加权 L1
        weighted_l1 = sum(lam * l1 for lam, l1 in zip(lambdas, l1_losses))

        # MAPE 平均
        avg_mape = sum(mape_losses) / len(mape_losses)
        weighted_mape = self.lambda_mape * avg_mape

        # 总损失
        total_loss = (
            self.lambda_cls * loss_cls
            + weighted_l1
            + weighted_mape
        )

        loss_dict = {
            "total": total_loss.item(),
            "cls": (self.lambda_cls * loss_cls).item(),
        }
        for i, name in enumerate(REGRESSION_NAMES):
            loss_dict[f"{name}_l1"] = (lambdas[i] * l1_losses[i]).item()
            loss_dict[f"{name}_mape"] = mape_losses[i].item()
        loss_dict["mape_avg"] = weighted_mape.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = MultiTaskLossV2(num_classes=61).to(device)

    logits = torch.randn(4, 61).to(device)
    labels = torch.tensor([0, 5, 10, 60]).to(device)
    pred = torch.tensor([
        [250, 200, 15, 30, 10],
        [180, 150, 12, 25, 8],
        [350, 300, 25, 40, 15],
        [120, 100, 8, 15, 5],
    ]).to(device).float()
    gt = torch.tensor([
        [230, 190, 14, 28, 12],
        [190, 160, 13, 26, 9],
        [340, 310, 24, 38, 14],
        [110, 95, 7, 14, 6],
    ]).to(device).float()

    total_loss, loss_dict = criterion(logits, labels, pred, gt)
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
