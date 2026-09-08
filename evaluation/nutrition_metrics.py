"""
营养估计评估指标
=================
评估食物卡路里和重量回归质量的MAPE、RMSE和MAE指标。

MAPE (Mean Absolute Percentage Error):
    平均绝对百分比误差，衡量相对误差
    MAPE = mean(|pred - gt| / |gt|) * 100%
    典型值: <20% 优秀, 20-50% 良好, >50% 较差

RMSE (Root Mean Square Error):
    均方根误差，对大误差更敏感
    RMSE = sqrt(mean((pred - gt)^2))

MAE (Mean Absolute Error):
    平均绝对误差，鲁棒性好的基本指标
    MAE = mean(|pred - gt|)
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def mape(pred: np.ndarray, target: np.ndarray, epsilon: float = 1.0) -> float:
    """计算MAPE

    Args:
        pred: 预测值数组
        target: 真实值数组
        epsilon: 防止除零的常数 (默认1.0, 适用于卡路里和重量)

    Returns:
        MAPE百分比
    """
    return float(np.mean(np.abs(pred - target) / (np.abs(target) + epsilon)) * 100)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """计算RMSE

    Args:
        pred: 预测值数组
        target: 真实值数组

    Returns:
        RMSE值
    """
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """计算MAE

    Args:
        pred: 预测值数组
        target: 真实值数组

    Returns:
        MAE值
    """
    return float(np.mean(np.abs(pred - target)))


class NutritionMetrics:
    """营养估计评估指标集合

    支持逐样本和批量计算。

    Args:
        calorie_unit: 卡路里单位 (默认 'kcal')
        weight_unit: 重量单位 (默认 'g')
    """

    def __init__(
        self,
        calorie_unit: str = "kcal",
        weight_unit: str = "g",
    ):
        self.calorie_unit = calorie_unit
        self.weight_unit = weight_unit

    def compute(
        self,
        pred_calories: np.ndarray,
        gt_calories: np.ndarray,
        pred_weights: Optional[np.ndarray] = None,
        gt_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """计算所有营养指标

        Args:
            pred_calories: 预测卡路里 [N]
            gt_calories: 真实卡路里 [N]
            pred_weights: 预测重量 [N] (可选)
            gt_weights: 真实重量 [N] (可选)

        Returns:
            dict: 各项指标
        """
        results = {}

        # 卡路里指标
        results["cal_mape"] = mape(pred_calories, gt_calories)
        results["cal_rmse"] = rmse(pred_calories, gt_calories)
        results["cal_mae"] = mae(pred_calories, gt_calories)

        # 重量指标（如果提供）
        if pred_weights is not None and gt_weights is not None:
            results["weight_mape"] = mape(pred_weights, gt_weights)
            results["weight_rmse"] = rmse(pred_weights, gt_weights)
            results["weight_mae"] = mae(pred_weights, gt_weights)

        return results

    def compute_from_tensors(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """从torch Tensor计算指标

        Args:
            pred: 预测值 [B, 2] (卡路里, 重量) 或 [B, 1] (仅卡路里)
            target: 真实值，同形状

        Returns:
            dict: 各项指标
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 2 and pred_np.shape[1] >= 2:
            return self.compute(
                pred_calories=pred_np[:, 0],
                gt_calories=target_np[:, 0],
                pred_weights=pred_np[:, 1],
                gt_weights=target_np[:, 1],
            )
        else:
            pred_flat = pred_np.flatten()
            target_flat = target_np.flatten()
            return self.compute(pred_flat, target_flat)

    def compute_per_category(
        self,
        pred_calories: np.ndarray,
        gt_calories: np.ndarray,
        categories: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """按食物类别分别计算指标

        Args:
            pred_calories: 预测卡路里 [N]
            gt_calories: 真实卡路里 [N]
            categories: 类别索引 [N]

        Returns:
            dict: {category_idx: {mape, rmse, mae}}
        """
        unique_cats = np.unique(categories)
        per_category = {}

        for cat in unique_cats:
            mask = categories == cat
            if mask.sum() > 0:
                per_category[int(cat)] = {
                    "mape": mape(pred_calories[mask], gt_calories[mask]),
                    "rmse": rmse(pred_calories[mask], gt_calories[mask]),
                    "mae": mae(pred_calories[mask], gt_calories[mask]),
                    "count": int(mask.sum()),
                }

        return per_category

    @staticmethod
    def format_results(results: Dict[str, float]) -> str:
        """格式化输出结果"""
        lines = ["营养估计评估结果:"]
        if "cal_mape" in results:
            lines.append(f"  卡路里 MAPE: {results['cal_mape']:.2f}%")
            lines.append(f"  卡路里 RMSE: {results['cal_rmse']:.2f} kcal")
            lines.append(f"  卡路里 MAE:  {results['cal_mae']:.2f} kcal")
        if "weight_mape" in results:
            lines.append(f"  重量 MAPE:   {results['weight_mape']:.2f}%")
            lines.append(f"  重量 RMSE:   {results['weight_rmse']:.2f} g")
            lines.append(f"  重量 MAE:    {results['weight_mae']:.2f} g")
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试营养指标
    np.random.seed(42)

    # 模拟预测和真值
    gt_cal = np.array([200, 350, 150, 500, 280, 420, 180, 300])
    pred_cal = gt_cal + np.random.randn(len(gt_cal)) * 30  # 添加噪声

    gt_weight = np.array([150, 250, 100, 350, 200, 300, 120, 220])
    pred_weight = gt_weight + np.random.randn(len(gt_weight)) * 20

    metrics = NutritionMetrics()
    results = metrics.compute(pred_cal, gt_cal, pred_weight, gt_weight)
    print(metrics.format_results(results))

    # 按类别测试
    categories = np.array([0, 1, 0, 2, 1, 2, 0, 1])
    per_cat = metrics.compute_per_category(pred_cal, gt_cal, categories)
    print("\n按类别统计:")
    for cat, vals in per_cat.items():
        print(f"  类别{cat}: MAPE={vals['mape']:.2f}%, n={vals['count']}")
