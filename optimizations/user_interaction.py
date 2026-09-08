"""
优化6: 用户交互 — 圈选/修正
============================
允许用户在界面上圈选食物区域、修正分类和营养估计结果，
将人工反馈融入系统提升最终精度。

思路:
    1. 圈选交互: 用户在图像上画框/多边形标出食物区域
       - 点击图像 → 获取坐标 → 生成交互mask
       - mask送入注意力模块或分割模块引导

    2. 修正交互: 用户修正错误的分类/卡路里估计
       - 选择正确类别 → 更新分类结果
       - 拖动卡路里滑块 → 调整回归预测
       - 反馈数据可收集用于在线学习

    3. 多食物场景: 一张图中有多种食物
       - 用户分别圈选 → 各区域独立分析
       - 各区域卡路里汇总 → 总卡路里

    4. 渐进式修正: 模型初始估计 → 用户微调 → 最终结果
       - 减少用户操作负担
       - 模型先给出最佳猜测，用户只需修正偏差

实现:
    本模块提供交互数据结构和修正逻辑，
    具体UI实现在app/gradio_demo.py中。
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class InteractionType(Enum):
    """交互类型枚举"""
    BBOX = "bbox"               # 矩形框选
    POLYGON = "polygon"         # 多边形圈选
    POINT = "point"             # 点击选择
    CATEGORY_FIX = "category_fix"   # 修正分类
    CALORIE_FIX = "calorie_fix"     # 修正卡路里
    WEIGHT_FIX = "weight_fix"       # 修正重量


@dataclass
class UserSelection:
    """用户选择区域

    Attributes:
        interaction_type: 交互类型
        points: 坐标点列表 [(x1,y1), (x2,y2), ...]
                bbox: [(x1,y1), (x2,y2)] 左上角和右下角
                polygon: [(x1,y1), (x2,y2), ...] 多边形顶点
                point: [(x,y)] 点击坐标
        label: 用户指定的标签 (可选)
        image_size: 原始图像尺寸 (H, W)
    """
    interaction_type: InteractionType
    points: List[Tuple[float, float]]
    label: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None

    def to_mask(self, height: int, width: int) -> np.ndarray:
        """将用户选择转换为二值mask

        Args:
            height: 图像高度
            width: 图像宽度

        Returns:
            二值mask [H, W], 1=选中区域, 0=背景
        """
        mask = np.zeros((height, width), dtype=np.float32)

        if self.interaction_type == InteractionType.BBOX:
            # 矩形框
            (x1, y1), (x2, y2) = self.points[0], self.points[1]
            # 坐标归一化处理
            if 0 < max(x1, x2) <= 1.0:
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
            x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
            mask[y1:y2, x1:x2] = 1.0

        elif self.interaction_type == InteractionType.POLYGON:
            # 多边形填充
            try:
                from PIL import Image, ImageDraw
                img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(img)
                # 归一化坐标处理
                poly_points = []
                for px, py in self.points:
                    if 0 < max(px, py) <= 1.0:
                        px, py = px * width, py * height
                    poly_points.append((int(px), int(py)))
                draw.polygon(poly_points, fill=255)
                mask = np.array(img).astype(np.float32) / 255.0
            except ImportError:
                # 无PIL时退化为bbox
                xs = [p[0] for p in self.points]
                ys = [p[1] for p in self.points]
                x1, x2 = int(min(xs) * width), int(max(xs) * width)
                y1, y2 = int(min(ys) * height), int(max(ys) * height)
                mask[y1:y2, x1:x2] = 1.0

        elif self.interaction_type == InteractionType.POINT:
            # 点击 → 生成小区域
            px, py = self.points[0]
            if 0 < px <= 1.0:
                px, py = int(px * width), int(py * height)
            radius = min(height, width) // 20
            y_start = max(0, int(py) - radius)
            y_end = min(height, int(py) + radius)
            x_start = max(0, int(px) - radius)
            x_end = min(width, int(px) + radius)
            mask[y_start:y_end, x_start:x_end] = 1.0

        return mask


@dataclass
class CorrectionFeedback:
    """用户修正反馈

    Attributes:
        original_category: 原始预测类别
        corrected_category: 修正后类别
        original_calories: 原始预测卡路里
        corrected_calories: 修正后卡路里
        original_weight: 原始预测重量
        corrected_weight: 修正后重量
    """
    original_category: Optional[str] = None
    corrected_category: Optional[str] = None
    original_calories: Optional[float] = None
    corrected_calories: Optional[float] = None
    original_weight: Optional[float] = None
    corrected_weight: Optional[float] = None

    def has_category_fix(self) -> bool:
        return (self.corrected_category is not None and
                self.corrected_category != self.original_category)

    def has_calorie_fix(self) -> bool:
        return (self.corrected_calories is not None and
                self.original_calories is not None and
                abs(self.corrected_calories - self.original_calories) > 1.0)

    def has_weight_fix(self) -> bool:
        return (self.corrected_weight is not None and
                self.original_weight is not None and
                abs(self.corrected_weight - self.original_weight) > 1.0)


class InteractionProcessor:
    """用户交互处理器

    处理用户圈选和修正，更新模型预测结果。

    Args:
        class_names: 食物类别名称列表
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def apply_selection(
        self,
        image: torch.Tensor,
        selections: List[UserSelection],
    ) -> Dict[str, Any]:
        """应用用户圈选，生成多区域mask

        Args:
            image: 原始图像 [B, 3, H, W] 或 [3, H, W]
            selections: 用户选择列表

        Returns:
            dict: {
                'combined_mask': [B, 1, H, W]  合并后的mask
                'region_masks':  List[Tensor]   各区域独立mask
                'num_regions':   int            区域数量
            }
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, _, H, W = image.shape

        combined_mask = torch.zeros(B, 1, H, W)
        region_masks = []

        for i, sel in enumerate(selections):
            mask_np = sel.to_mask(H, W)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            region_masks.append(mask_tensor.expand(B, -1, -1, -1))
            combined_mask = torch.max(combined_mask, mask_tensor.expand(B, -1, -1, -1))

        return {
            "combined_mask": combined_mask,
            "region_masks": region_masks,
            "num_regions": len(selections),
        }

    def apply_correction(
        self,
        predictions: Dict[str, Any],
        feedback: CorrectionFeedback,
    ) -> Dict[str, Any]:
        """应用用户修正到模型预测

        Args:
            predictions: 原始预测 {
                'category': str, 'category_prob': float,
                'calories': float, 'weight': float
            }
            feedback: 用户修正

        Returns:
            修正后的预测
        """
        corrected = predictions.copy()

        if feedback.has_category_fix():
            corrected["category"] = feedback.corrected_category
            corrected["category_corrected"] = True

        if feedback.has_calorie_fix():
            corrected["calories"] = feedback.corrected_calories
            corrected["calories_corrected"] = True

        if feedback.has_weight_fix():
            corrected["weight"] = feedback.corrected_weight
            corrected["weight_corrected"] = True

        return corrected

    def process_multi_food_image(
        self,
        image: torch.Tensor,
        selections: List[UserSelection],
    ) -> List[Dict[str, torch.Tensor]]:
        """处理多食物图像

        对用户圈选的每个区域，裁剪并准备单独分析。

        Args:
            image: 原始图像 [3, H, W]
            selections: 各食物区域的用户选择

        Returns:
            各区域的裁剪图像和mask列表
        """
        results = []
        _, H, W = image.shape

        for sel in selections:
            mask = sel.to_mask(H, W)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

            # 裁剪到mask的bounding box
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # 加padding
                pad = 10
                rmin = max(0, rmin - pad)
                rmax = min(H, rmax + pad)
                cmin = max(0, cmin - pad)
                cmax = min(W, cmax + pad)

                cropped = image[:, rmin:rmax, cmin:cmax]
                cropped_mask = mask_tensor[:, rmin:rmax, cmin:cmax]
            else:
                cropped = image
                cropped_mask = mask_tensor

            results.append({
                "image": cropped,
                "mask": cropped_mask,
                "label": sel.label,
            })

        return results


if __name__ == "__main__":
    # 测试用户选择
    print("=== 测试UserSelection ===")

    # 矩形框选
    bbox = UserSelection(
        interaction_type=InteractionType.BBOX,
        points=[(0.1, 0.2), (0.8, 0.9)],
        image_size=(256, 256),
    )
    mask = bbox.to_mask(256, 256)
    print(f"BBox mask: shape={mask.shape}, 非零像素={mask.sum():.0f}")

    # 点击选择
    point = UserSelection(
        interaction_type=InteractionType.POINT,
        points=[(128, 128)],
    )
    mask = point.to_mask(256, 256)
    print(f"Point mask: shape={mask.shape}, 非零像素={mask.sum():.0f}")

    # 测试修正反馈
    print("\n=== 测试CorrectionFeedback ===")
    feedback = CorrectionFeedback(
        original_category="苹果",
        corrected_category="梨",
        original_calories=95.0,
        corrected_calories=100.0,
    )
    print(f"分类修正: {feedback.has_category_fix()}")
    print(f"卡路里修正: {feedback.has_calorie_fix()}")

    # 测试交互处理器
    print("\n=== 测试InteractionProcessor ===")
    processor = InteractionProcessor(class_names=["苹果", "梨", "香蕉"])
    image = torch.randn(3, 256, 256)
    selections = [bbox]
    result = processor.apply_selection(image, selections)
    print(f"合并mask: {result['combined_mask'].shape}")
    print(f"区域数: {result['num_regions']}")
