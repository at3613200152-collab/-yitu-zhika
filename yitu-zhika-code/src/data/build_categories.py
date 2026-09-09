"""
食物类别构建脚本
==================
从 dish_ingredients.csv 提取每道菜的主要食材，
基于关键词映射推导食物类别，生成带 category 列的新 dishes.csv。

用法:
    python build_categories.py --data_dir data/Nutrition5k --output data/Nutrition5k/dishes_categorized.csv
"""

import os
import csv
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ============================================================
# 食材关键词 → 类别映射
# 按优先级排列，越靠前优先级越高（如 "chicken breast" 先匹配 meat）
# 注意：P1-A 改为「词边界 + 右端/最长短语优先」，不再用裸子串 `kw in name`，
#       避免 eggplant→egg、graham cracker→meat(ham)、cream of mushroom soup→dairy(cream) 这类误匹配。
# ============================================================
CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("meat", [
        "pork", "beef", "chicken", "bacon", "steak", "turkey", "ham",
        "sausage", "lamb", "meatball", "meat", "veal", "duck", "ribs",
        "brisket", "pulled pork", "ground beef", "chorizo", "pepperoni",
        "salami", "prosciutto", "pancetta", "ground turkey", "ground pork",
    ]),
    ("seafood", [
        "fish", "shrimp", "salmon", "tuna", "crab", "lobster", "cod",
        "tilapia", "scallop", "clam", "mussel", "oyster", "squid",
        "octopus", "sardine", "anchovy", "catfish", "prawn",
    ]),
    ("egg", [
        "egg", "scrambled egg", "fried egg", "omelet", "omelette",
    ]),
    ("dairy", [
        "cheese", "yogurt", "milk", "cream", "butter", "cottage cheese",
        "mozzarella", "parmesan", "feta", "ricotta", "cheddar",
        "cream cheese", "sour cream", "whipped cream", "ice cream",
        "buttermilk",
    ]),
    ("grain", [
        "rice", "pasta", "bread", "noodle", "quinoa", "cereal", "oats",
        "flour", "tortilla", "wrap", "bagel", "baguette", "crouton",
        "couscous", "polenta", "grits", "pancake", "waffle", "croissant",
        "bun", "roll", "cracker", "pretzel", "cornmeal", "flatbread",
    ]),
    ("soup_stew", [
        "soup", "broth", "stew", "chili", "curry", "chowder",
        "gumbo", "bisque",
    ]),
    ("vegetable", [
        "salad", "asparagus", "cauliflower", "broccoli", "spinach",
        "mixed greens", "onions", "peppers", "tomato", "carrot",
        "mushroom", "corn", "potato", "sweet potato", "zucchini",
        "eggplant", "cucumber", "cabbage", "lettuce", "kale",
        "brussels sprouts", "green beans", "peas", "celery",
        "avocado", "olives", "salsa", "coleslaw", "garden salad",
        "caesar salad", "vinegar", "parsley", "lemon juice", "lime",
        "garlic", "ginger", "herbs", "basil", "cilantro", "mint",
        "scallion", "shallot", "leek", "artichoke", "radish",
        "beets", "turnip", "pumpkin", "squash", "arugula",
        "watercress", "endive", "radicchio", "fennel", "lemon",
        "apple", "berries", "strawberries", "fruit", "banana",
        "orange", "grape", "melon", "pineapple", "mango", "peach",
        "pear", "cherry", "blueberry", "raspberry", "cranberry",
        "raisin", "dried fruit", "date", "fig",
        # 不规则复数/常见复合（词边界+s/es 无法覆盖，显式列出）
        "chickpea", "blackberries", "blueberries", "raspberries", "cranberries",
    ]),
    ("dessert", [
        "cake", "cookie", "chocolate", "brownie", "pie", "tart",
        "pudding", "muffin", "donut", "doughnut", "pastry",
        "candy", "caramel", "fudge", "gelatin", "jello",
        "sugar", "honey", "maple syrup", "jam", "jelly",
    ]),
    ("sauce_condiment", [
        "soy sauce", "hot sauce", "ketchup", "mustard", "mayonnaise",
        "dressing", "sauce", "syrup", "vinegar", "oil", "salt",
        "pepper", "spice", "seasoning", "marinade", "gravy",
        "aioli", "tahini", "sriracha", "worcestershire",
    ]),
]

# 人工覆盖：已知容易误判的整段食材/菜名（优先级最高，精确或短语匹配）。
# 保留命中原因，供审计与后续人工核验。
OVERRIDES: Dict[str, Tuple[str, str]] = {
    "apple pie": ("dessert", "override"),
    "apple crisp": ("dessert", "override"),
    "graham cracker": ("grain", "override"),
    "graham crackers": ("grain", "override"),
    "eggplant": ("vegetable", "override"),
    "cream of mushroom soup": ("soup_stew", "override"),
    "cream of chicken soup": ("soup_stew", "override"),
    "cream soup": ("soup_stew", "override"),
    "mushroom soup": ("soup_stew", "override"),
    "ice cream": ("dessert", "override"),
    "sour cream": ("dairy", "override"),
    "whipped cream": ("dairy", "override"),
    "cream cheese": ("dairy", "override"),
    "sweet potato": ("vegetable", "override"),
    "caesar salad": ("vegetable", "override"),
    "garden salad": ("vegetable", "override"),
}


def _keyword_boundary_pattern(kw: str) -> str:
    # 词边界：关键词前后不能紧贴其它字母（避免 egg→eggplant、ham→graham）。
    # 允许关键词后跟复数后缀 s/es（eggs、cucumbers、cookies 等仍能命中）。
    r = re.escape(kw)
    return r"(?<![a-z])" + r + r"(?:s|es)?(?![a-z])"


def match_ingredient(ingr_name: str) -> Tuple[str, str]:
    """返回 (category, matched_keyword)。matched_keyword 供审计/人工核验。

    匹配策略（P1-A）：
      1) OVERRIDES 精确/短语优先；
      2) 词边界（\b）避免子串误匹配；
      3) 右端优先（食材/菜名的"头名词"往往靠后：apple pie→pie、cream of mushroom soup→soup）；
      4) 同一位次下再取最长短语，最后按类别列表优先级兜底。
    """
    name = ingr_name.lower().strip()
    if name in OVERRIDES:
        cat, reason = OVERRIDES[name]
        return cat, reason

    best = None  # (start, kw_len, -category_order, category, kw)
    for cat_order, (category, keywords) in enumerate(CATEGORY_KEYWORDS):
        for kw in keywords:
            m = re.search(_keyword_boundary_pattern(kw), name)
            if m:
                score = (m.start(), len(kw), -cat_order)
                if best is None or score > best[0]:
                    best = (score, category, kw)
    if best is None:
        return "other", None
    return best[1], best[2]


def classify_ingredient(ingr_name: str) -> str:
    """根据食材名称推断类别（内部用 match_ingredient）。"""
    return match_ingredient(ingr_name)[0]


# 旧规则（仅审计对比用）：裸子串 + 类别优先级
def classify_ingredient_legacy(ingr_name: str) -> str:
    name_lower = ingr_name.lower().strip()
    for category, keywords in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"


def build_dish_categories(data_dir: str) -> Dict[str, str]:
    """从 dish_ingredients.csv 构建每道菜的类别

    策略：按重量(g)加权统计各类别占比，取占比最高的类别。
    如果最高占比 < 40%，则标记为 "mixed"。
    """
    dish_ingredients_path = os.path.join(data_dir, "dish_ingredients.csv")

    if not os.path.exists(dish_ingredients_path):
        print(f"  ⚠️ dish_ingredients.csv not found at {dish_ingredients_path}")
        return {}

    # 每道菜的类别重量统计
    dish_category_weights: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    with open(dish_ingredients_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_id = row.get("dish_id", "")
            ingr_name = row.get("ingr_name", "")
            try:
                grams = float(row.get("grams", 0))
            except (ValueError, TypeError):
                grams = 0.0

            if not dish_id or not ingr_name or grams <= 0:
                continue

            category = classify_ingredient(ingr_name)
            dish_category_weights[dish_id][category] += grams

    # 确定每道菜的类别
    dish_categories: Dict[str, str] = {}

    for dish_id, cat_weights in dish_category_weights.items():
        total_weight = sum(cat_weights.values())
        if total_weight <= 0:
            dish_categories[dish_id] = "other"
            continue

        # 找出占比最高的类别
        best_cat = max(cat_weights, key=cat_weights.get)
        best_ratio = cat_weights[best_cat] / total_weight

        if best_ratio >= 0.40:
            dish_categories[dish_id] = best_cat
        else:
            dish_categories[dish_id] = "mixed"

    return dish_categories


def main():
    parser = argparse.ArgumentParser(description="构建食物类别标签")
    parser.add_argument("--data_dir", type=str, default="data/Nutrition5k",
                        help="Nutrition5k数据目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出CSV路径（默认覆盖dishes.csv）")
    args = parser.parse_args()

    print(f"数据目录: {args.data_dir}")

    # 构建类别映射
    print("从dish_ingredients.csv推导食物类别...")
    dish_categories = build_dish_categories(args.data_dir)
    print(f"  推导了 {len(dish_categories)} 道菜的类别")

    # 统计类别分布
    cat_dist = defaultdict(int)
    for cat in dish_categories.values():
        cat_dist[cat] += 1
    print("\n类别分布:")
    for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:4d} 道 ({count/len(dish_categories)*100:.1f}%)")

    # 读取原始dishes.csv并添加category列
    dishes_path = os.path.join(args.data_dir, "dishes.csv")
    output_path = args.output or os.path.join(args.data_dir, "dishes_categorized.csv")

    with open(dishes_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames)

        # 如果已有category列就替换，没有就添加
        if "category" not in fieldnames:
            fieldnames.append("category")

        rows = []
        for row in reader:
            dish_id = row.get("dish_id", "")
            row["category"] = dish_categories.get(dish_id, "other")
            rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n已生成: {output_path} ({len(rows)} 行)")
    print(f"类别数: {len(cat_dist)}")


if __name__ == "__main__":
    main()
