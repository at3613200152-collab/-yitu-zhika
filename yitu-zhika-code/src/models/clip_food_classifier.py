"""
CLIP食物分类器
================
使用CLIP视觉-语言模型进行零样本食物分类。
不需要训练，直接用预训练CLIP模型识别图片中的食物。

工作流程:
    1. 加载ingredients.csv构建食物文本列表
    2. CLIP编码图片 + 所有食物文本 → 相似度排序
    3. 返回top-k食物及置信度

使用:
    classifier = CLIPFoodClassifier()
    result = classifier.classify("path/to/food_image.jpg")
    # result = [{"name": "white rice", "confidence": 0.85, ...}, ...]
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional


# CLIP食物分类的默认配置
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TOP_K = 5
MAX_INGREDIENTS = 200  # 限制最大食材数量，避免过多


class CLIPFoodClassifier:
    """CLIP零样本食物分类器"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ingredients_csv: str = None,
        device: str = None,
        max_ingredients: int = MAX_INGREDIENTS,
    ):
        """
        Args:
            model_name: HuggingFace模型名
            ingredients_csv: ingredients.csv路径（含食材名和每克营养）
            device: 计算设备
            max_ingredients: 最大食材数量（按出现频率截取）
        """
        from transformers import CLIPProcessor, CLIPModel

        # 设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"CLIP设备: {device}")

        # 加载CLIP模型
        print(f"加载CLIP模型: {model_name} ...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("CLIP模型加载完成")

        # 加载食材数据库
        self.ingredients = self._load_ingredients(ingredients_csv, max_ingredients)
        print(f"加载了 {len(self.ingredients)} 个食材类别")

        # 预编码所有文本（加速推理）
        self._text_features = None
        self._encode_texts()

    def _load_ingredients(self, csv_path: str, max_count: int) -> List[Dict]:
        """加载食材数据库"""
        if csv_path is None:
            # 尝试默认路径
            candidates = [
                "data/Nutrition5k/ingredients.csv",
                "./data/Nutrition5k/ingredients.csv",
                os.path.join(os.path.dirname(__file__), "..", "..", "data",
                             "Nutrition5k", "ingredients.csv"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    csv_path = p
                    break

        if csv_path is None or not os.path.exists(csv_path):
            print("⚠️ 未找到ingredients.csv，使用默认食物列表")
            return self._default_foods()

        ingredients = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("ingr", "").strip()
                if not name:
                    continue
                try:
                    ingredients.append({
                        "name": name,
                        "cal_per_g": float(row.get("cal/g", 0)),
                        "fat_per_g": float(row.get("fat(g)", 0)),
                        "carb_per_g": float(row.get("carb(g)", 0)),
                        "protein_per_g": float(row.get("protein(g)", 0)),
                    })
                except (ValueError, TypeError):
                    continue

        # 如果太多，取前max_count个
        if len(ingredients) > max_count:
            ingredients = ingredients[:max_count]

        return ingredients if ingredients else self._default_foods()

    def _default_foods(self) -> List[Dict]:
        """默认食物列表（当CSV不可用时）"""
        default = [
            ("white rice", 1.19, 0.004, 0.25, 0.05),
            ("brown rice", 1.19, 0.004, 0.25, 0.05),
            ("grilled chicken", 1.65, 0.05, 0.0, 0.31),
            ("beef steak", 2.71, 0.19, 0.0, 0.25),
            ("pork", 2.43, 0.14, 0.0, 0.27),
            ("salad", 0.65, 0.034, 0.032, 0.061),
            ("pasta", 1.31, 0.012, 0.25, 0.05),
            ("bread", 2.65, 0.033, 0.49, 0.09),
            ("pizza", 2.66, 0.094, 0.335, 0.112),
            ("fish", 2.06, 0.13, 0.0, 0.22),
            ("shrimp", 0.99, 0.001, 0.0, 0.24),
            ("egg", 1.48, 0.11, 0.016, 0.1),
            ("cheese", 3.5, 0.33, 0.013, 0.25),
            ("potatoes", 0.77, 0.001, 0.17, 0.02),
            ("soup", 0.4, 0.01, 0.05, 0.02),
            ("noodles", 1.38, 0.02, 0.25, 0.05),
            ("burger", 2.5, 0.15, 0.25, 0.15),
            ("sandwich", 2.3, 0.12, 0.28, 0.12),
            ("vegetables", 0.3, 0.003, 0.06, 0.015),
            ("fruit", 0.5, 0.003, 0.12, 0.01),
        ]
        return [
            {"name": n, "cal_per_g": c, "fat_per_g": f, "carb_per_g": cb, "protein_per_g": p}
            for n, c, f, cb, p in default
        ]

    def _encode_texts(self):
        """预编码所有食物文本prompt"""
        prompts = [f"a photo of {ing['name']}" for ing in self.ingredients]
        print(f"编码 {len(prompts)} 个食物文本...")

        with torch.no_grad():
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(self.device)
            self._text_features = self.model.get_text_features(**inputs)
            # 归一化
            self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True)
        print("文本编码完成")

    def classify(self, image: Image.Image or str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """对食物图片进行零样本分类

        Args:
            image: PIL.Image或图片路径
            top_k: 返回前k个结果

        Returns:
            [{"name": "white rice", "confidence": 0.85,
              "cal_per_g": 1.19, "fat_per_g": 0.004, ...}, ...]
        """
        # 加载图片
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        with torch.no_grad():
            # 编码图片
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算与所有食物文本的相似度
            similarity = (image_features @ self._text_features.T).squeeze(0)
            # softmax得到概率
            probs = similarity.softmax(dim=-1).cpu().numpy()

        # 取top_k
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            ing = self.ingredients[idx]
            results.append({
                "name": ing["name"],
                "confidence": float(probs[idx]),
                "cal_per_g": ing["cal_per_g"],
                "fat_per_g": ing["fat_per_g"],
                "carb_per_g": ing["carb_per_g"],
                "protein_per_g": ing["protein_per_g"],
            })

        return results

    def get_weighted_nutrition(self, classify_results: List[Dict]) -> Dict[str, float]:
        """根据CLIP分类结果加权平均营养值

        Args:
            classify_results: classify()的返回值

        Returns:
            {"cal_per_g": 1.5, "fat_per_g": 0.05, "carb_per_g": 0.2, "protein_per_g": 0.15}
        """
        if not classify_results:
            return {"cal_per_g": 1.5, "fat_per_g": 0.05, "carb_per_g": 0.2, "protein_per_g": 0.15}

        total_weight = sum(r["confidence"] for r in classify_results)
        if total_weight <= 0:
            total_weight = 1.0

        weighted = {}
        for key in ["cal_per_g", "fat_per_g", "carb_per_g", "protein_per_g"]:
            weighted[key] = sum(r[key] * r["confidence"] for r in classify_results) / total_weight

        return weighted


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    if img_path is None:
        print("用法: python clip_food_classifier.py <image_path>")
        sys.exit(1)

    classifier = CLIPFoodClassifier()
    results = classifier.classify(img_path, top_k=5)

    print("\nCLIP食物分类结果:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:30s} 置信度={r['confidence']:.4f} "
              f"cal/g={r['cal_per_g']:.3f}")

    weighted = classifier.get_weighted_nutrition(results)
    print(f"\n加权平均每克营养:")
    print(f"  热量: {weighted['cal_per_g']:.3f} cal/g")
    print(f"  蛋白质: {weighted['protein_per_g']:.3f} g/g")
    print(f"  碳水: {weighted['carb_per_g']:.3f} g/g")
    print(f"  脂肪: {weighted['fat_per_g']:.3f} g/g")
