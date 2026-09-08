"""
食物营养数据库
==============
整合USDA和中国食物成分表数据，内置50+中国常见食物的营养数据。

数据格式:
    {菜名: {
        'calories': 热量(kcal/100g),
        'protein': 蛋白质(g/100g),
        'carb': 碳水化合物(g/100g),
        'fat': 脂肪(g/100g),
        'fiber': 膳食纤维(g/100g),
        'category': 分类 (主食/肉禽/水产/蔬菜/水果/豆制品/蛋奶/饮品/小吃/调味)
    }}

数据来源:
    - 中国食物成分表(第6版)
    - USDA FoodData Central
    - 常见菜品营养数据取均值估计

注意: 数据为每100g可食部的营养含量，实际菜品可能因烹饪方式有差异。
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class FoodItem:
    """食物营养项"""
    name: str               # 食物名称
    calories: float         # 热量 (kcal/100g)
    protein: float          # 蛋白质 (g/100g)
    carb: float             # 碳水化合物 (g/100g)
    fat: float              # 脂肪 (g/100g)
    fiber: float            # 膳食纤维 (g/100g)
    category: str           # 分类

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calories": self.calories,
            "protein": self.protein,
            "carb": self.carb,
            "fat": self.fat,
            "fiber": self.fiber,
            "category": self.category,
        }


# ============================================================
# 中国常见食物营养数据字典 (50+种)
# ============================================================
FOOD_DATABASE: Dict[str, Dict[str, Any]] = {
    # ==================== 主食 ====================
    "白米饭": {"calories": 116, "protein": 2.6, "carb": 25.9, "fat": 0.3, "fiber": 0.3, "category": "主食"},
    "糙米饭": {"calories": 111, "protein": 2.8, "carb": 23.5, "fat": 0.9, "fiber": 1.8, "category": "主食"},
    "馒头": {"calories": 221, "protein": 7.0, "carb": 47.0, "fat": 1.1, "fiber": 1.3, "category": "主食"},
    "面条(煮)": {"calories": 110, "protein": 3.4, "carb": 24.3, "fat": 0.4, "fiber": 0.8, "category": "主食"},
    "饺子(猪肉)": {"calories": 215, "protein": 8.2, "carb": 25.0, "fat": 8.5, "fiber": 1.0, "category": "主食"},
    "包子(猪肉)": {"calories": 227, "protein": 7.2, "carb": 28.0, "fat": 9.0, "fiber": 0.8, "category": "主食"},
    "油条": {"calories": 386, "protein": 6.9, "carb": 51.0, "fat": 17.6, "fiber": 0.9, "category": "主食"},
    "小米粥": {"calories": 46, "protein": 1.4, "carb": 8.4, "fat": 0.7, "fiber": 0.2, "category": "主食"},
    "炒饭": {"calories": 175, "protein": 4.5, "carb": 26.0, "fat": 6.0, "fiber": 0.5, "category": "主食"},
    "煎饼": {"calories": 235, "protein": 6.5, "carb": 38.0, "fat": 8.0, "fiber": 1.5, "category": "主食"},

    # ==================== 肉禽 ====================
    "红烧肉": {"calories": 337, "protein": 13.2, "carb": 3.5, "fat": 30.8, "fiber": 0.0, "category": "肉禽"},
    "糖醋排骨": {"calories": 265, "protein": 15.0, "carb": 12.0, "fat": 18.0, "fiber": 0.2, "category": "肉禽"},
    "宫保鸡丁": {"calories": 197, "protein": 18.0, "carb": 8.5, "fat": 10.5, "fiber": 1.0, "category": "肉禽"},
    "白切鸡": {"calories": 167, "protein": 22.0, "carb": 2.5, "fat": 7.5, "fiber": 0.0, "category": "肉禽"},
    "回锅肉": {"calories": 310, "protein": 14.0, "carb": 5.0, "fat": 27.0, "fiber": 0.5, "category": "肉禽"},
    "鱼香肉丝": {"calories": 185, "protein": 16.0, "carb": 10.0, "fat": 9.0, "fiber": 1.5, "category": "肉禽"},
    "京酱肉丝": {"calories": 220, "protein": 17.0, "carb": 8.0, "fat": 13.5, "fiber": 0.5, "category": "肉禽"},
    "水煮牛肉": {"calories": 145, "protein": 20.0, "carb": 3.0, "fat": 6.0, "fiber": 0.5, "category": "肉禽"},
    "烤鸡翅": {"calories": 222, "protein": 18.0, "carb": 4.0, "fat": 15.0, "fiber": 0.0, "category": "肉禽"},
    "东坡肉": {"calories": 380, "protein": 12.0, "carb": 5.0, "fat": 35.0, "fiber": 0.0, "category": "肉禽"},

    # ==================== 水产 ====================
    "清蒸鲈鱼": {"calories": 105, "protein": 19.0, "carb": 0.5, "fat": 3.5, "fiber": 0.0, "category": "水产"},
    "红烧鱼": {"calories": 145, "protein": 17.0, "carb": 5.0, "fat": 6.5, "fiber": 0.2, "category": "水产"},
    "水煮鱼": {"calories": 130, "protein": 17.5, "carb": 3.5, "fat": 5.5, "fiber": 0.3, "category": "水产"},
    "蒜蓉虾": {"calories": 93, "protein": 18.0, "carb": 2.0, "fat": 1.8, "fiber": 0.0, "category": "水产"},
    "红烧带鱼": {"calories": 160, "protein": 16.0, "carb": 5.5, "fat": 8.5, "fiber": 0.1, "category": "水产"},
    "酸菜鱼": {"calories": 115, "protein": 18.0, "carb": 4.0, "fat": 3.5, "fiber": 0.5, "category": "水产"},
    "油焖大虾": {"calories": 168, "protein": 17.0, "carb": 5.0, "fat": 8.0, "fiber": 0.0, "category": "水产"},

    # ==================== 蔬菜 ====================
    "清炒西兰花": {"calories": 35, "protein": 3.0, "carb": 4.3, "fat": 0.6, "fiber": 2.5, "category": "蔬菜"},
    "醋溜白菜": {"calories": 42, "protein": 1.5, "carb": 5.0, "fat": 1.8, "fiber": 1.2, "category": "蔬菜"},
    "蒜蓉空心菜": {"calories": 38, "protein": 2.5, "carb": 4.0, "fat": 1.2, "fiber": 1.5, "category": "蔬菜"},
    "干煸四季豆": {"calories": 85, "protein": 3.0, "carb": 6.0, "fat": 5.5, "fiber": 2.0, "category": "蔬菜"},
    "地三鲜": {"calories": 120, "protein": 3.5, "carb": 12.0, "fat": 6.5, "fiber": 2.0, "category": "蔬菜"},
    "麻婆豆腐": {"calories": 95, "protein": 6.5, "carb": 4.5, "fat": 5.5, "fiber": 0.5, "category": "蔬菜"},
    "西红柿炒蛋": {"calories": 86, "protein": 5.5, "carb": 5.0, "fat": 5.0, "fiber": 0.8, "category": "蔬菜"},
    "清炒菠菜": {"calories": 30, "protein": 2.8, "carb": 3.5, "fat": 0.5, "fiber": 1.8, "category": "蔬菜"},
    "土豆丝": {"calories": 75, "protein": 2.5, "carb": 14.0, "fat": 1.5, "fiber": 1.0, "category": "蔬菜"},
    "红烧茄子": {"calories": 80, "protein": 1.5, "carb": 8.0, "fat": 4.5, "fiber": 1.5, "category": "蔬菜"},

    # ==================== 水果 ====================
    "苹果": {"calories": 53, "protein": 0.2, "carb": 13.7, "fat": 0.1, "fiber": 1.7, "category": "水果"},
    "香蕉": {"calories": 93, "protein": 1.4, "carb": 22.2, "fat": 0.2, "fiber": 1.2, "category": "水果"},
    "橙子": {"calories": 48, "protein": 0.8, "carb": 11.1, "fat": 0.2, "fiber": 0.6, "category": "水果"},
    "葡萄": {"calories": 44, "protein": 0.5, "carb": 10.3, "fat": 0.2, "fiber": 0.4, "category": "水果"},
    "西瓜": {"calories": 25, "protein": 0.5, "carb": 5.8, "fat": 0.1, "fiber": 0.3, "category": "水果"},

    # ==================== 豆制品 ====================
    "豆腐": {"calories": 73, "protein": 7.0, "carb": 2.8, "fat": 3.5, "fiber": 0.4, "category": "豆制品"},
    "豆浆": {"calories": 16, "protein": 1.8, "carb": 1.1, "fat": 0.7, "fiber": 0.1, "category": "豆制品"},
    "腐竹": {"calories": 461, "protein": 44.6, "carb": 22.3, "fat": 21.7, "fiber": 1.0, "category": "豆制品"},

    # ==================== 蛋奶 ====================
    "煮鸡蛋": {"calories": 144, "protein": 13.3, "carb": 1.5, "fat": 9.5, "fiber": 0.0, "category": "蛋奶"},
    "牛奶": {"calories": 54, "protein": 3.0, "carb": 3.4, "fat": 3.2, "fiber": 0.0, "category": "蛋奶"},
    "酸奶": {"calories": 72, "protein": 3.1, "carb": 9.3, "fat": 2.7, "fiber": 0.0, "category": "蛋奶"},

    # ==================== 汤品 ====================
    "紫菜蛋花汤": {"calories": 18, "protein": 1.5, "carb": 1.5, "fat": 0.8, "fiber": 0.2, "category": "汤品"},
    "番茄蛋汤": {"calories": 25, "protein": 1.8, "carb": 2.5, "fat": 1.0, "fiber": 0.3, "category": "汤品"},
    "排骨莲藕汤": {"calories": 95, "protein": 6.0, "carb": 5.5, "fat": 5.5, "fiber": 0.8, "category": "汤品"},
    "鸡汤": {"calories": 55, "protein": 4.5, "carb": 1.5, "fat": 3.5, "fiber": 0.0, "category": "汤品"},

    # ==================== 饮品 ====================
    "绿茶": {"calories": 1, "protein": 0.0, "carb": 0.3, "fat": 0.0, "fiber": 0.0, "category": "饮品"},
    "豆浆(甜)": {"calories": 33, "protein": 2.0, "carb": 4.5, "fat": 0.8, "fiber": 0.2, "category": "饮品"},
    "可乐": {"calories": 43, "protein": 0.0, "carb": 10.6, "fat": 0.0, "fiber": 0.0, "category": "饮品"},

    # ==================== 小吃 ====================
    "春卷(炸)": {"calories": 265, "protein": 5.0, "carb": 25.0, "fat": 15.0, "fiber": 1.0, "category": "小吃"},
    "烧麦": {"calories": 238, "protein": 8.0, "carb": 22.0, "fat": 12.5, "fiber": 0.5, "category": "小吃"},
    "葱油饼": {"calories": 310, "protein": 6.0, "carb": 38.0, "fat": 14.0, "fiber": 1.0, "category": "小吃"},
    "煎饺": {"calories": 250, "protein": 7.5, "carb": 24.0, "fat": 12.0, "fiber": 0.8, "category": "小吃"},
}


class FoodNutritionDB:
    """食物营养数据库

    提供食物营养查询、搜索和推荐功能。

    Args:
        custom_data: 自定义食物数据 (可选，合并到默认数据库)
    """

    def __init__(self, custom_data: Optional[Dict] = None):
        self.data = dict(FOOD_DATABASE)
        if custom_data:
            self.data.update(custom_data)
        self._items = {name: FoodItem(name=name, **info) for name, info in self.data.items()}

    def get(self, name: str) -> Optional[FoodItem]:
        """按名称查询食物"""
        return self._items.get(name)

    def get_dict(self, name: str) -> Optional[Dict]:
        """按名称查询食物（返回字典）"""
        item = self.get(name)
        return item.to_dict() if item else None

    def search(self, keyword: str) -> List[FoodItem]:
        """按关键词搜索食物"""
        results = []
        for name, item in self._items.items():
            if keyword in name or keyword in item.category:
                results.append(item)
        return results

    def list_by_category(self, category: str) -> List[FoodItem]:
        """按分类列出食物"""
        return [item for item in self._items.values() if item.category == category]

    def list_all_categories(self) -> List[str]:
        """列出所有分类"""
        return sorted(set(item.category for item in self._items.values()))

    def get_all_names(self) -> List[str]:
        """获取所有食物名称"""
        return list(self._items.keys())

    def calculate_meal_nutrition(
        self,
        meal: Dict[str, float],
    ) -> Dict[str, float]:
        """计算一餐的总营养

        Args:
            meal: {食物名: 重量(g)}

        Returns:
            总营养 {calories, protein, carb, fat, fiber}
        """
        total = {"calories": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0, "fiber": 0.0}

        for food_name, grams in meal.items():
            item = self.get(food_name)
            if item is None:
                continue
            ratio = grams / 100.0
            total["calories"] += item.calories * ratio
            total["protein"] += item.protein * ratio
            total["carb"] += item.carb * ratio
            total["fat"] += item.fat * ratio
            total["fiber"] += item.fiber * ratio

        return total

    def add_food(self, name: str, nutrition: Dict[str, Any]):
        """添加自定义食物"""
        self.data[name] = nutrition
        self._items[name] = FoodItem(name=name, **nutrition)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


if __name__ == "__main__":
    db = FoodNutritionDB()
    print(f"数据库包含 {len(db)} 种食物")
    print(f"分类: {db.list_all_categories()}")

    # 查询示例
    food = db.get("宫保鸡丁")
    if food:
        print(f"\n{food.name}: {food.calories} kcal/100g")
        print(f"  蛋白质: {food.protein}g, 碳水: {food.carb}g, 脂肪: {food.fat}g")

    # 搜索
    results = db.search("鸡")
    print(f"\n搜索'鸡': {len(results)}个结果")
    for r in results:
        print(f"  {r.name}: {r.calories} kcal")

    # 计算一餐营养
    meal = {"白米饭": 200, "宫保鸡丁": 150, "清炒西兰花": 100}
    nutrition = db.calculate_meal_nutrition(meal)
    print(f"\n一餐营养: {nutrition}")
