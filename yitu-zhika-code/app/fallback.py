"""ImageNet + 知识库容错降级。

模型推理失败时按优先级降级：
  1. RGB + NIR 模型 → 失败
  2. 纯 RGB 模型 → 失败
  3. ImageNet 分类 + 11 类映射 + 知识库均值 → 兜底
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 11 个粗类别在 Nutrition5k 上的均值（基于 dishes_verified.csv 统计）
# 当所有模型都失败时，用类别均值兜底，明确标注 fallback
CATEGORY_MEAN_KCAL = {
    "dairy": 280.0, "dessert": 350.0, "egg": 180.0,
    "grain": 320.0, "meat": 250.0, "mixed": 300.0,
    "other": 250.0, "sauce_condiment": 150.0,
    "seafood": 220.0, "soup_stew": 180.0, "vegetable": 150.0,
}

CATEGORY_MEAN_MASS = {
    "dairy": 200.0, "dessert": 180.0, "egg": 150.0,
    "grain": 250.0, "meat": 200.0, "mixed": 280.0,
    "other": 220.0, "sauce_condiment": 100.0,
    "seafood": 200.0, "soup_stew": 300.0, "vegetable": 200.0,
}

# ImageNet 1000 类 → 11 粗类别的映射（关键食物相关类）
IMAGENET_TO_CATEGORY = {
    "pizza": "grain", "pizza_dough": "grain",
    "plate": "mixed", "tray": "mixed",
    "cheese": "dairy", "milk": "dairy", "cream": "dairy", "yogurt": "dairy",
    "ice_cream": "dessert", "cake": "dessert", "doughnut": "dessert",
    "bagel": "grain", "bread": "grain", "pretzel": "grain", "muffin": "grain",
    "hotpot": "soup_stew", "soup": "soup_stew", "consomme": "soup_stew",
    "steak": "meat", "meat_loaf": "meat", "bacon": "meat", "hotdog": "meat",
    "brisket": "meat", "ribs": "meat", "hamburger": "meat",
    "guacamole": "vegetable", "salad": "vegetable", "broccoli": "vegetable",
    "cauliflower": "vegetable", "cucumber": "vegetable", "artichoke": "vegetable",
    "shellfish": "seafood", "lobster": "seafood", "crab": "seafood",
    "shrimp": "seafood", "oyster": "seafood",
    "syrup": "sauce_condiment", "ketchup": "sauce_condiment",
    "mayonnaise": "sauce_condiment", "mashed_potato": "vegetable",
    " french_loaf": "grain", "baguet": "grain",
    "french_fries": "vegetable", "french_onion_soup": "soup_stew",
}


def fallback_to_imagenet(imagenet_top5: list[str], classifier=None) -> dict | None:
    """ImageNet 分类 top5 → 11 类映射。

    参数：
      imagenet_top5: ImageNet 类名列表（如 ['pizza', 'plate', ...]）
      classifier: 可选的 CLIP 分类器，None 时用 ImageNet 映射
    返回：{category_name, source} 或 None（无法映射时）
    """
    if classifier is not None:
        # 走 CLIP 11 类分类器（如果已加载）
        try:
            return {"category_name": classifier(imagenet_top5), "source": "clip"}
        except Exception:
            pass

    for label in imagenet_top5:
        cat = IMAGENET_TO_CATEGORY.get(label.lower().replace(" ", "_"))
        if cat:
            return {"category_name": cat, "source": "imagenet_map"}
    return None


def fallback_to_knowledge_base(category: str | None) -> dict:
    """用知识库类别均值兜底。明确标注 fallback_kb。"""
    if category is None or category not in CATEGORY_MEAN_KCAL:
        category = "mixed"
    return {
        "calories_kcal": CATEGORY_MEAN_KCAL[category],
        "weight_g": CATEGORY_MEAN_MASS[category],
        "category_name": category,
        "status": "fallback_kb",
        "source": "knowledge_base_mean",
        "warning": "模型推理失败，使用类别均值兜底；不应用于饮食或医疗决策。",
    }


def safe_predict(pipeline, image) -> dict:
    """带容错的预测。任何模型失败时降级，不返回 None。"""
    from app.contracts import PredictionContract, InferenceStatus
    try:
        result = pipeline.predict(image)
        contract = PredictionContract.from_pipeline_result(
            result, "meal_nir_official_v1")
        # 检查负值
        if contract.calories_kcal < 0 or contract.weight_g < 0:
            contract.warnings.append("出现负值，模型在此图上失效")
            contract.status = InferenceStatus.MODEL_ERROR.value
        return contract.to_dict()
    except Exception as e:
        # 完全失败，走知识库兜底
        kb = fallback_to_knowledge_base(None)
        kb["error"] = str(e)
        return kb