"""
多任务损失函数
==============
包含分类损失(CrossEntropy)和回归损失(L1 + MAPE)的加权组合。

总损失:
    L = λ_cls * CrossEntropy(logits, labels)
      + λ_cal * L1(cal_pred, cal_gt)
      + λ_weight * L1(weight_pred, weight_gt)
      + λ_mape * (MAPE(cal) + MAPE(weight))

MAPE (Mean Absolute Percentage Error):
    MAPE = mean(|pred - gt| / |gt|) * 100%
    对营养估计尤为重要，因为它衡量的是相对误差而非绝对误差。
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class MAPELoss(nn.Module):
    """MAPE (Mean Absolute Percentage Error) 损失

    MAPE = mean(|pred - target| / (|target| + eps)) * 100

    对零值或接近零值的目标需要加epsilon防止除零。

    Args:
        epsilon: 防止除零的小常数 (默认1.0)
        reduction: 归约方式 ('mean' / 'sum' / 'none')
    """

    def __init__(self, epsilon: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算MAPE

        Args:
            pred: 预测值 [B, ...]
            target: 目标值 [B, ...] (必须非负)

        Returns:
            MAPE标量
        """
        # 防止除零：对target加epsilon（卡路里和重量通常>0，epsilon=1.0影响很小）
        denom = torch.abs(target) + self.epsilon
        mape = torch.abs(pred - target) / denom

        if self.reduction == "mean":
            return mape.mean()
        elif self.reduction == "sum":
            return mape.sum()
        else:
            return mape


class MultiTaskLoss(nn.Module):
    """多任务综合损失

    L = λ_cls * CE_loss + λ_cal * L1_cal + λ_weight * L1_weight + λ_mape * MAPE_total

    Args:
        num_classes: 分类类别数（用于label smoothing）
        lambda_cls: 分类损失权重
        lambda_cal: 卡路里回归损失权重
        lambda_weight: 重量回归损失权重
        lambda_mape: MAPE损失权重
        label_smoothing: 标签平滑系数 (默认0.1)
        mape_epsilon: MAPE中的防除零常数
        log_transform: 回归目标是否为log1p空间。True时L1在log空间计算,
                       MAPE先expm1还原到物理空间(kcal/g)再计算相对误差
    """

    def __init__(
        self,
        num_classes: int = 61,
        lambda_cls: float = 1.0,
        lambda_cal: float = 1.0,
        lambda_weight: float = 0.5,
        lambda_mape: float = 0.1,
        label_smoothing: float = 0.1,
        mape_epsilon: float = 1.0,
        log_transform: bool = False,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_cal = lambda_cal
        self.lambda_weight = lambda_weight
        self.lambda_mape = lambda_mape
        self.log_transform = log_transform

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
            logits: 分类logits [B, num_classes]
            labels: 分类标签 [B]
            nutrition_pred: 回归预测 [B, 2] (卡路里, 重量)
            nutrition_gt: 回归真值 [B, 2] (卡路里, 重量)

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失明细
        """
        # 分类损失
        loss_cls = self.criterion_cls(logits, labels)

        # 卡路里回归L1损失
        cal_pred = nutrition_pred[:, 0]
        cal_gt = nutrition_gt[:, 0]
        loss_cal_l1 = self.criterion_l1(cal_pred, cal_gt)

        # 重量回归L1损失
        weight_pred = nutrition_pred[:, 1]
        weight_gt = nutrition_gt[:, 1]
        loss_weight_l1 = self.criterion_l1(weight_pred, weight_gt)

        # MAPE损失（卡路里和重量合计）
        # log_transform时目标在log1p空间: 先expm1还原到物理空间(kcal/g)再算相对误差,
        # 否则MAPE的"百分比"语义不成立; clamp(max=20)防止expm1数值溢出
        if self.log_transform:
            cal_p = torch.expm1(torch.clamp(cal_pred, max=20.0)).clamp(min=0.0)
            cal_g = torch.expm1(cal_gt)
            w_p = torch.expm1(torch.clamp(weight_pred, max=20.0)).clamp(min=0.0)
            w_g = torch.expm1(weight_gt)
        else:
            cal_p, cal_g = cal_pred, cal_gt
            w_p, w_g = weight_pred, weight_gt
        loss_mape_cal = self.criterion_mape(cal_p, cal_g)
        loss_mape_weight = self.criterion_mape(w_p, w_g)
        loss_mape = (loss_mape_cal + loss_mape_weight) * 0.5

        # 总损失
        total_loss = (
            self.lambda_cls * loss_cls
            + self.lambda_cal * loss_cal_l1
            + self.lambda_weight * loss_weight_l1
            + self.lambda_mape * loss_mape
        )

        loss_dict = {
            "total": total_loss.item(),
            "cls": (self.lambda_cls * loss_cls).item(),
            "cal_l1": (self.lambda_cal * loss_cal_l1).item(),
            "weight_l1": (self.lambda_weight * loss_weight_l1).item(),
            "mape_cal": loss_mape_cal.item(),
            "mape_weight": loss_mape_weight.item(),
            "mape": (self.lambda_mape * loss_mape).item(),
        }
        # 原物理尺度的MAE(供训练监控直读: kcal / g)
        with torch.no_grad():
            loss_dict["mae_cal"] = (cal_p - cal_g).abs().mean().item()
            loss_dict["mae_mass"] = (w_p - w_g).abs().mean().item()

        return total_loss, loss_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试MAPE损失
    mape = MAPELoss(epsilon=1.0)
    pred = torch.tensor([200.0, 150.0, 300.0])
    target = torch.tensor([180.0, 160.0, 310.0])
    print(f"MAPE: {mape(pred, target).item():.2f}%")

    # 测试多任务损失
    criterion = MultiTaskLoss(
        num_classes=61,
        lambda_cls=1.0, lambda_cal=1.0,
        lambda_weight=0.5, lambda_mape=0.1,
    ).to(device)

    logits = torch.randn(4, 61).to(device)
    labels = torch.tensor([0, 5, 10, 60]).to(device)
    nutrition_pred = torch.tensor([[250.0, 200.0], [180.0, 150.0],
                                   [350.0, 300.0], [120.0, 100.0]]).to(device)
    nutrition_gt = torch.tensor([[230.0, 190.0], [190.0, 160.0],
                                  [340.0, 310.0], [110.0, 95.0]]).to(device)

    total_loss, loss_dict = criterion(logits, labels, nutrition_pred, nutrition_gt)
    print(f"多任务损失: {loss_dict}")
