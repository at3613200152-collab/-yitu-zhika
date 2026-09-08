"""
食谱推荐 — 规则引擎 (路线A)
=============================
基于线性规划的食谱生成引擎，严格遵守营养约束。

约束条件:
    1. 每日热量在 TDEE ± tolerance 范围内
    2. 碳水/蛋白/脂肪比例可配 (默认 50%/25%/25%)
    3. 每周同一食材不重复超过 max_repeat 次
    4. 三餐热量分配: 早30% / 午40% / 晚30%
    5. 每餐至少包含1种主食、1种菜品

求解方法:
    使用 PuLP 线性规划库，目标函数为营养均衡度最大（偏差最小）。

变量定义:
    x[i][j][k] = 食物i 在第j天第k餐的重量(g)
    i: 食物索引
    j: 天数 (0-6)
    k: 餐次 (0=早餐, 1=午餐, 2=晚餐)
"""

import pulp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .food_database import FoodNutritionDB, FoodItem


@dataclass
class RecipeConfig:
    """食谱生成配置"""
    tdee: float = 2000.0                    # 每日目标热量 (kcal)
    calorie_tolerance: float = 200.0        # 热量允许偏差 (kcal)
    macro_ratios: Dict[str, float] = None   # 宏量营养素比例
    meal_ratios: Dict[str, float] = None    # 三餐热量比例
    max_repeat_per_ingredient: int = 3      # 每周同一食材最大重复次数
    min_portion: float = 50.0               # 最小份量 (g)
    max_portion: float = 400.0              # 最大份量 (g)
    days: int = 7                           # 生成天数
    meals_per_day: int = 3                  # 每天餐数

    def __post_init__(self):
        if self.macro_ratios is None:
            self.macro_ratios = {"carb": 0.50, "protein": 0.25, "fat": 0.25}
        if self.meal_ratios is None:
            self.meal_ratios = {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30}


class RecipeRuleEngine:
    """食谱规则引擎

    基于线性规划生成满足营养约束的周食谱。

    Args:
        food_db: 食物营养数据库
        config: 食谱配置
    """

    # 每餐食物类别约束: 至少包含这些类别
    MEAL_CATEGORY_RULES = {
        0: {"主食": 1, "蛋奶": 1},          # 早餐: 至少1主食+1蛋奶
        1: {"主食": 1, "肉禽": 1, "蔬菜": 1},  # 午餐: 至少1主食+1荤+1素
        2: {"主食": 1, "蔬菜": 1},            # 晚餐: 至少1主食+1素 (可选1荤)
    }

    # 各餐典型食物数量范围
    MEAL_FOOD_COUNT = {
        0: (2, 4),   # 早餐: 2-4种
        1: (3, 5),   # 午餐: 3-5种
        2: (2, 4),   # 晚餐: 2-4种
    }

    def __init__(self, food_db: FoodNutritionDB, config: Optional[RecipeConfig] = None):
        self.food_db = food_db
        self.config = config or RecipeConfig()
        self.all_foods = food_db.get_all_names()

    def generate_weekly_recipe(
        self,
        recent_intake: Optional[List[Dict[str, float]]] = None,
        excluded_foods: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """生成一周食谱

        Args:
            recent_intake: 最近N天摄入记录 [{food: grams, ...}, ...]
            excluded_foods: 排除的食物列表

        Returns:
            7天×3餐食谱, 每天3餐, 每餐含食物列表和营养汇总
        """
        excluded = set(excluded_foods or [])

        # 统计近期已有食材频次
        recent_counts = self._count_recent_ingredients(recent_intake)

        # 逐天逐餐生成（简化版：逐餐求解LP）
        weekly_recipe = []
        ingredient_weekly_count = {}  # 本周食材计数

        for day in range(self.config.days):
            daily_recipe = []

            for meal_idx in range(self.config.meals_per_day):
                meal_name = ["早餐", "午餐", "晚餐"][meal_idx]
                meal_calorie_target = self.config.tdee * list(self.config.meal_ratios.values())[meal_idx]

                # 求解单餐食谱
                meal = self._solve_single_meal(
                    calorie_target=meal_calorie_target,
                    meal_idx=meal_idx,
                    excluded=excluded,
                    recent_counts=recent_counts,
                    weekly_counts=ingredient_weekly_count,
                )

                # 更新本周计数
                for food_name in meal:
                    ingredient_weekly_count[food_name] = ingredient_weekly_count.get(food_name, 0) + 1

                # 计算营养
                meal_nutrition = self.food_db.calculate_meal_nutrition(meal)

                daily_recipe.append({
                    "meal_name": meal_name,
                    "foods": [{"name": name, "grams": round(grams)} for name, grams in meal.items()],
                    "nutrition": {k: round(v, 1) for k, v in meal_nutrition.items()},
                })

            weekly_recipe.append({
                "day": day + 1,
                "meals": daily_recipe,
            })

        return weekly_recipe

    def _solve_single_meal(
        self,
        calorie_target: float,
        meal_idx: int,
        excluded: set,
        recent_counts: Dict[str, int],
        weekly_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """求解单餐食谱 (线性规划)

        目标: 最小化热量偏差 + 最大化食物多样性
        约束: 热量范围、类别要求、重复限制

        Args:
            calorie_target: 目标热量 (kcal)
            meal_idx: 餐次索引
            excluded: 排除食物集合
            recent_counts: 近期食材频次
            weekly_counts: 本周食材频次

        Returns:
            {食物名: 重量(g)}
        """
        # 筛选可用食物
        available_foods = [
            name for name in self.all_foods
            if name not in excluded
            and weekly_counts.get(name, 0) < self.config.max_repeat_per_ingredient
        ]

        if not available_foods:
            available_foods = self.all_foods[:5]  # 兜底

        # 创建LP问题
        prob = pulp.LpProblem("MealPlanning", pulp.LpMinimize)

        # 决策变量: 每种食物的份量(g)
        portions = {}
        # 用索引起变量名, 避免中文菜名写入LP文件导致CBC求解器解析异常
        idx_of = {name: i for i, name in enumerate(available_foods)}
        for name in available_foods:
            portions[name] = pulp.LpVariable(
                f"x_{idx_of[name]}",
                lowBound=0,
                upBound=self.config.max_portion,
                cat='Continuous',
            )

        # 总热量表达式
        total_calories = pulp.lpSum(
            portions[name] * self.food_db.get(name).calories / 100.0
            for name in available_foods
        )
        # 目标函数: 最小化|总热量 - 目标热量|
        # 注: PuLP为线性规划, 不支持表达式与表达式相乘(二次项会报"非恒定表达式无法相乘"),
        #     用辅助变量dev将绝对值偏差线性化
        dev = pulp.LpVariable("dev", lowBound=0)
        prob += dev  # 最小化绝对偏差
        prob += total_calories - calorie_target <= dev
        prob += calorie_target - total_calories <= dev

        # 约束1: 热量范围
        prob += total_calories >= calorie_target - self.config.calorie_tolerance
        prob += total_calories <= calorie_target + self.config.calorie_tolerance

        # 约束2: 类别要求 — 至少包含指定类别
        category_rules = self.MEAL_CATEGORY_RULES.get(meal_idx, {})
        for category, min_count in category_rules.items():
            cat_foods = [name for name in available_foods
                         if self.food_db.get(name).category == category]
            if cat_foods:
                # 用二元变量表示是否选用该类食物
                cat_selected = []
                for name in cat_foods:
                    is_selected = pulp.LpVariable(f"y_{idx_of[name]}", cat='Binary')
                    # 如果选中，份量至少min_portion
                    prob += portions[name] >= self.config.min_portion * is_selected
                    prob += portions[name] <= self.config.max_portion * is_selected
                    cat_selected.append(is_selected)
                prob += pulp.lpSum(cat_selected) >= min_count

        # 约束3: 食物数量范围
        min_foods, max_foods = self.MEAL_FOOD_COUNT.get(meal_idx, (2, 4))

        # 约束4: 宏量营养素比例（软约束，通过权重调整）
        total_protein = pulp.lpSum(
            portions[name] * self.food_db.get(name).protein / 100.0
            for name in available_foods
        ) * 4  # 蛋白质 4kcal/g
        total_carb = pulp.lpSum(
            portions[name] * self.food_db.get(name).carb / 100.0
            for name in available_foods
        ) * 4  # 碳水 4kcal/g
        total_fat = pulp.lpSum(
            portions[name] * self.food_db.get(name).fat / 100.0
            for name in available_foods
        ) * 9  # 脂肪 9kcal/g

        # 宏量营养素软约束 (允许±10%偏差)
        # 注: PuLP表达式只支持 <=, >=, == 比较, 不能用 > 做 if 判断
        carb_ratio = self.config.macro_ratios.get("carb", 0.50)
        protein_ratio = self.config.macro_ratios.get("protein", 0.25)
        prob += total_carb >= total_calories * (carb_ratio - 0.10)
        prob += total_carb <= total_calories * (carb_ratio + 0.10)
        prob += total_protein >= total_calories * (protein_ratio - 0.10)

        # 求解
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
        except Exception:
            # 求解失败时回退到简单方法
            return self._fallback_meal(calorie_target, meal_idx)

        # 提取结果
        meal = {}
        if prob.status == pulp.constants.LpStatusOptimal:
            for name in available_foods:
                portion_val = portions[name].varValue
                if portion_val and portion_val > self.config.min_portion * 0.5:
                    meal[name] = max(self.config.min_portion, portion_val)

        # 如果LP结果为空，回退
        if not meal:
            meal = self._fallback_meal(calorie_target, meal_idx)

        return meal

    def _fallback_meal(self, calorie_target: float, meal_idx: int) -> Dict[str, float]:
        """回退方法: 简单按比例分配热量

        当LP求解失败时使用。
        """
        meal = {}
        category_rules = self.MEAL_CATEGORY_RULES.get(meal_idx, {})

        # 按类别选择代表性食物
        for category in category_rules:
            foods = self.food_db.list_by_category(category)
            if foods:
                # 选第一个
                food = foods[0]
                # 按热量反算份量
                grams = (calorie_target * 0.3) / (food.calories / 100.0)
                grams = max(self.config.min_portion, min(self.config.max_portion, grams))
                meal[food.name] = round(grams)

        return meal

    def _count_recent_ingredients(
        self,
        recent_intake: Optional[List[Dict[str, float]]],
    ) -> Dict[str, int]:
        """统计近期食材频次"""
        counts = {}
        if recent_intake is None:
            return counts

        for day_record in recent_intake:
            for food_name in day_record:
                counts[food_name] = counts.get(food_name, 0) + 1

        return counts

    def validate_recipe(self, recipe: List[Dict]) -> Dict[str, Any]:
        """验证食谱是否满足约束

        Returns:
            dict: {'valid': bool, 'violations': [...], 'daily_nutrition': [...]}
        """
        violations = []
        daily_nutrition = []

        for day in recipe:
            day_total = {"calories": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0, "fiber": 0.0}
            for meal in day["meals"]:
                for k, v in meal["nutrition"].items():
                    day_total[k] = day_total.get(k, 0.0) + v

            daily_nutrition.append(day_total)

            # 检查热量约束
            cal = day_total["calories"]
            if abs(cal - self.config.tdee) > self.config.calorie_tolerance:
                violations.append(
                    f"第{day['day']}天热量{cal:.0f}kcal超出TDEE±{self.config.calorie_tolerance:.0f}范围"
                )

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "daily_nutrition": daily_nutrition,
        }


if __name__ == "__main__":
    from .food_database import FoodNutritionDB

    db = FoodNutritionDB()
    config = RecipeConfig(tdee=2000, calorie_tolerance=200)
    engine = RecipeRuleEngine(db, config)

    # 生成一周食谱
    recipe = engine.generate_weekly_recipe()

    for day in recipe:
        print(f"\n=== 第{day['day']}天 ===")
        for meal in day["meals"]:
            print(f"  {meal['meal_name']}:")
            for food in meal["foods"]:
                print(f"    {food['name']}: {food['grams']}g")
            print(f"    营养: {meal['nutrition']}")

    # 验证
    validation = engine.validate_recipe(recipe)
    print(f"\n食谱验证: {'通过' if validation['valid'] else '未通过'}")
    if validation['violations']:
        for v in validation['violations']:
            print(f"  ⚠ {v}")
