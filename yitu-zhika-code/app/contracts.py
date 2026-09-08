"""模型服务契约。统一单位、版本、错误状态规范。

所有 inference 输出必须遵循此契约，便于小程序和 Demo 一致消费。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelVersion(str, Enum):
    """已审计的模型版本。"""
    RGB_V1 = "meal_rgb_official_v1"
    NIR_V1 = "meal_nir_official_v1"
    EXTERNAL_V1 = "calorieclip_official_v1"


class InferenceStatus(str, Enum):
    """推理结果状态。"""
    OK = "ok"                       # 正常推理
    FALLBACK_RGB = "fallback_rgb"   # NIR 失败，降级到纯 RGB
    FALLBACK_KB = "fallback_kb"     # 模型全失败，降级到知识库
    INVALID_IMAGE = "invalid_image" # 输入图片无效
    MODEL_ERROR = "model_error"     # 模型内部错误


@dataclass
class PredictionContract:
    """统一预测结果契约。所有数字带单位字符串，避免歧义。"""
    # 核心字段
    calories_kcal: float        # 总热量，单位 kcal
    weight_g: float             # 总重量，单位 g
    category_name: str          # 中文粗类别名（11 类之一）
    category_prob: float        # 模型分数，未校准

    # 模型版本溯源
    model_version: str          # ModelVersion 值
    inference_precision: str    # "FP32" / "BF16"
    device: str                 # "cpu" / "cuda:0"

    # 状态
    status: str = InferenceStatus.OK.value
    warnings: list = field(default_factory=list)

    # 对照字段（可选）
    rgb_calories_kcal: float | None = None
    rgb_weight_g: float | None = None
    external_calories_kcal: float | None = None
    external_status: str | None = None  # 外部基线状态描述

    # NIR 图（不序列化，仅 Demo 用）
    nir_image_b64: str | None = None

    @classmethod
    def from_pipeline_result(cls, result: dict, model_version: str) -> "PredictionContract":
        """从 ExperimentPipeline.predict() 结果构造契约对象。"""
        return cls(
            calories_kcal=float(result["calories"]),
            weight_g=float(result["weight"]),
            category_name=result["category_name"],
            category_prob=float(result["category_prob"]),
            model_version=model_version,
            inference_precision=result.get("inference_precision", "FP32"),
            device=result.get("device", "cpu"),
            status=InferenceStatus.OK.value,
            rgb_calories_kcal=float(result.get("rgb_calories", 0)) or None,
            rgb_weight_g=float(result.get("rgb_weight", 0)) or None,
            external_calories_kcal=result.get("external_calories"),
            external_status=result.get("external_status"),
        )

    def to_dict(self) -> dict:
        """转 JSON 友好的 dict，去掉 None 字段。"""
        d = {
            "calories_kcal": round(self.calories_kcal, 1),
            "weight_g": round(self.weight_g, 1),
            "category_name": self.category_name,
            "category_prob": round(self.category_prob, 4),
            "model_version": self.model_version,
            "inference_precision": self.inference_precision,
            "device": self.device,
            "status": self.status,
        }
        if self.warnings:
            d["warnings"] = self.warnings
        if self.rgb_calories_kcal is not None:
            d["rgb_calories_kcal"] = round(self.rgb_calories_kcal, 1)
            d["rgb_weight_g"] = round(self.rgb_weight_g, 1)
        if self.external_calories_kcal is not None:
            d["external_calories_kcal"] = round(self.external_calories_kcal, 1)
        if self.external_status:
            d["external_status"] = self.external_status
        return d