"""食物前置过滤器：用 ImageNet 预训练 ResNet50 判断图片是否为食物。

为什么需要这个模块？
    主多任务网络（ResNet50，11 类食物 softmax）对域外数据（卡通、
    动漫、人脸、风景等）会强行分类为某一类食物并给出高置信度
    （甚至 1.0），这是 softmax 的固有缺陷——无法表达"都不是"。

工作原理：
    1. 用 ImageNet 预训练 ResNet50（1000 类，覆盖大量真实食物类别）
       对输入图片做 top-5 预测
    2. 如果 top-5 中没有任何食物/餐具/饮料类别，判定为非食物
    3. 如果 top-1 是食物且概率 >= 阈值，或 top-5 中食物概率总和
       超过阈值，通过

优点：
    - 无需训练，立即可用
    - 对卡通/动漫/人脸等域外数据鲁棒
    - ImageNet 类别覆盖真实食物（pizza/sushi/banana/...），
      比专用 11 类 softmax 更可靠
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

# ImageNet 1000 类中与食物/饮料/餐具相关的类别名关键词
# 参考: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
FOOD_KEYWORDS = {
    # 水果
    "banana", "lemon", "orange", "pineapple", "pomegranate", "fig", "strawberry",
    "apple", "custard", "ice lolly", "eggnog",
    # 蔬菜
    "cauliflower", "bell pepper", "cardoon", "mushroom", "zucchini", "cucumber",
    "artichoke", "spaghetti squash", "acorn squash", "butternut squash",
    "broccoli", "head cabbage", "corn",
    # 主食/菜品
    "guacamole", "hotdog", "hot dog", "french fries", "mashed potato",
    "chocolate sauce", "trifle", "hot pot", "cheeseburger", "spaghetti",
    "french loaf", "meat loaf", "carbonara", "pizza", "pretzel", "burrito",
    "bagel", "pretzel", "dough", "meatloaf",
    # 海鲜/肉
    "lobster", "crayfish", "crab", "shrimp", "oyster",
    "steak", "pork", "bacon", "ribs", "veal", "lamb",
    # 蛋/奶酪
    "cheese", "egg",
    # 餐具/容器/饮料（与食物强相关）
    "plate", "bowl", "tray", "cup", "espresso", "teapot", "goblet",
    "beer glass", "drinking glass", "water bottle", "bottle", "consomme",
    "red wine", "white wine", "wine", "ramen", "noodle", "soup",
    "ice cream", "sundae", "cake", "pie", "cookie", "brownie", "dessert",
}


def _build_food_class_indices() -> set:
    """从 ImageNet 类别列表中筛选食物相关类别的索引"""
    try:
        categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception as e:
        logger.warning(f"无法获取 ImageNet 类别列表: {e}，使用空集合")
        return set()

    food_indices = set()
    for idx, name in enumerate(categories):
        name_lower = name.lower()
        # 类别名包含任一食物关键词 → 视为食物类别
        if any(kw in name_lower for kw in FOOD_KEYWORDS):
            food_indices.add(idx)

    logger.info(f"ImageNet 食物相关类别: {len(food_indices)} 个")
    return food_indices


class FoodFilter:
    """食物前置过滤器：用 ImageNet 预训练 ResNet50 判断图片是否为食物。"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        # 用新 API 加载预训练权重
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(device)

        # ImageNet 标准预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 食物类别索引集合（动态构建）
        self.food_indices = _build_food_class_indices()
        # 缓存类别名
        try:
            self._categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        except Exception:
            self._categories = None

        logger.info(
            f"FoodFilter initialized: ResNet50 ImageNet, "
            f"{len(self.food_indices)} food-related classes"
        )

    def predict_top5(self, image: Image.Image) -> List[Tuple[int, float, str]]:
        """返回 top-5 预测 [(class_idx, prob, class_name), ...]"""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            top5_prob, top5_idx = probs.topk(5)

        result = []
        for i in range(5):
            idx = top5_idx[i].item()
            prob = top5_prob[i].item()
            result.append((idx, prob, self._class_name(idx)))
        return result

    def is_food(
        self,
        image: Image.Image,
        top5_threshold: float = 0.10,
        top1_food_threshold: float = 0.30,
    ) -> Tuple[bool, dict]:
        """判断图片是否为食物。

        Args:
            image: PIL Image
            top5_threshold: top-5 中食物类别概率总和阈值
            top1_food_threshold: top-1 是食物类别的概率阈值

        Returns:
            (is_food: bool, info: dict)
            info 包含 top5 预测、food_score、reason
        """
        top5 = self.predict_top5(image)
        food_score = sum(p for idx, p, _ in top5 if idx in self.food_indices)
        top1_idx, top1_prob, top1_name = top5[0]
        top1_is_food = top1_idx in self.food_indices

        # 判定逻辑：
        # 1. top-1 是食物且概率 >= 0.3 → 通过
        # 2. top-5 中食物概率总和 >= 0.1 → 通过（餐盘/食物组合）
        # 3. 否则拒绝
        is_food = (top1_is_food and top1_prob >= top1_food_threshold) or \
                  (food_score >= top5_threshold)

        reason = "top1_food" if (top1_is_food and top1_prob >= top1_food_threshold) else \
                 "top5_food_score" if food_score >= top5_threshold else "not_food"

        info = {
            "top5": [{"idx": idx, "prob": round(p, 4), "name": name}
                     for idx, p, name in top5],
            "food_score": round(food_score, 4),
            "top1_name": top1_name,
            "top1_prob": round(top1_prob, 4),
            "reason": reason,
        }
        return is_food, info

    def _class_name(self, idx: int) -> str:
        """获取 ImageNet 类别名"""
        if self._categories is not None and 0 <= idx < len(self._categories):
            return self._categories[idx]
        return f"class_{idx}"


# 模块单例（懒加载，避免重复加载模型权重）
_filter_instance: FoodFilter | None = None


def get_food_filter() -> FoodFilter:
    """获取 FoodFilter 单例"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = FoodFilter()
    return _filter_instance
