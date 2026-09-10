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
    category_name: str          # 中文粗类别名（label_schema 决定类别数：v1=11 类，v2=12 类含水果）
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

    # 溯源（P0-A）：主结果来源模型与 SHA，避免"版本=RGB 但数值来自 NIR"错标
    source_model_version: str | None = None
    source_model_sha256_prefix: str | None = None
    rgb_model_version: str | None = None
    target_names: list | None = None

    # 三大营养素（P0-A）：当前两目标模型不输出；值为 None + 状态，绝不伪造
    protein_g: float | None = None
    carbohydrate_g: float | None = None
    fat_g: float | None = None
    macros_status: str = "unsupported"    # supported / partial / unsupported
    macros_unit: str = "g"
    macros_source: str = "model_not_supported"
    category_prob_note: str = "模型置信度，非识别准确率"

    # 标签体系溯源（label_schema v2）：类别集合可能随版本变化，需与权重同步（11 类 / 12 类）
    label_schema: str | None = None
    category_manifest: str | None = None

    # 决策 C（2026-09-10）：热量/宏量与类别解耦，类别来自独立类别模型（12 类含水果）
    category_model: str | None = None
    category_model_sha256_prefix: str | None = None
    category_label_schema: str | None = None

    # NIR 图（不序列化，仅 Demo 用）
    nir_image_b64: str | None = None

    @classmethod
    def from_pipeline_result(cls, result: dict, model_version: str, model_sha_prefix: str | None = None) -> "PredictionContract":
        """从 ExperimentPipeline.predict() 结果构造契约对象。"""
        return cls(
            calories_kcal=float(result["calories"]),
            weight_g=float(result["weight"]),
            category_name=result["category_name"],
            category_prob=float(result["category_prob"]),
            model_version=model_version,
            source_model_version=result.get("source_model", model_version),
            source_model_sha256_prefix=model_sha_prefix or (result.get("source_model_sha256") or "")[:16],
            rgb_model_version=result.get("rgb_model"),
            target_names=result.get("target_names", ["calories", "mass"]),
            macros_status=result.get("macros_status", "unsupported"),
            macros_unit=result.get("macros_unit", "g"),
            macros_source=result.get("macros_source", "model_not_supported"),
            category_prob_note=result.get("category_prob_note", "模型置信度，非识别准确率"),
            label_schema=result.get("label_schema"),
            category_manifest=result.get("category_manifest"),
            category_model=result.get("category_model"),
            category_model_sha256_prefix=(result.get("category_model_sha256") or "")[:16] or None,
            category_label_schema=result.get("category_label_schema"),
            inference_precision=result.get("inference_precision", "FP32"),
            device=result.get("device", "cpu"),
            status=InferenceStatus.OK.value,
            rgb_calories_kcal=float(result.get("rgb_calories", 0)) or None,
            rgb_weight_g=float(result.get("rgb_weight", 0)) or None,
            external_calories_kcal=result.get("external_calories"),
            external_status=result.get("external_status"),
        )

    def to_dict(self) -> dict:
        """转 JSON 友好的 dict，去掉 None 的对照字段；宏量营养保留（可为 None + 状态）。"""
        d = {
            "calories_kcal": round(self.calories_kcal, 1),
            "weight_g": round(self.weight_g, 1),
            "category_name": self.category_name,
            "category_prob": round(self.category_prob, 3),
            "category_prob_note": self.category_prob_note,
            "model_version": self.model_version,
            "source_model_version": self.source_model_version,
            "source_model_sha256_prefix": self.source_model_sha256_prefix,
            "rgb_model_version": self.rgb_model_version,
            "target_names": self.target_names,
            "label_schema": self.label_schema,
            "category_manifest": self.category_manifest,
            "category_model": self.category_model,
            "category_model_sha256_prefix": self.category_model_sha256_prefix,
            "category_label_schema": self.category_label_schema,
            "macros": {
                "status": self.macros_status,
                "unit": self.macros_unit,
                "source": self.macros_source,
                "protein_g": self.protein_g,
                "carbohydrate_g": self.carbohydrate_g,
                "fat_g": self.fat_g,
            },
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