"""
周食谱生成器
==============
整合路线A(规则引擎)和路线B(LLM润色)，生成完整的周食谱。

流程:
    1. 用户信息 → TDEE估算
    2. TDEE + 营养偏好 → 规则引擎生成食谱骨架
    3. 食谱骨架 → LLM润色(做法/口味/贴士)
    4. 输出完整周食谱(含营养表+做法描述)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .food_database import FoodNutritionDB
from .rule_engine import RecipeRuleEngine, RecipeConfig
from .llm_polish import LLMPolisher
from .tdee_estimator import TDEEEstimator, UserProfile


class WeeklyPlanner:
    """周食谱生成器

    整合TDEE估算、规则引擎和LLM润色，一站式生成周食谱。

    Args:
        food_db: 食物营养数据库
        llm_backend: LLM后端类型
        llm_model: LLM模型名
        llm_api_key: LLM API密钥
    """

    def __init__(
        self,
        food_db: Optional[FoodNutritionDB] = None,
        llm_backend: str = "local",
        llm_model: str = "gpt-4",
        llm_api_key: Optional[str] = None,
    ):
        self.food_db = food_db or FoodNutritionDB()
        self.tdee_estimator = TDEEEstimator()
        self.llm_polisher = LLMPolisher(
            backend=llm_backend,
            model=llm_model,
            api_key=llm_api_key,
        )

    def generate(
        self,
        user_profile: UserProfile,
        recent_intake: Optional[List[Dict[str, float]]] = None,
        excluded_foods: Optional[List[str]] = None,
        use_llm_polish: bool = True,
        calorie_tolerance: float = 200.0,
        macro_ratios: Optional[Dict[str, float]] = None,
        meal_ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """生成完整周食谱

        Args:
            user_profile: 用户档案
            recent_intake: 近期摄入记录
            excluded_foods: 排除的食物
            use_llm_polish: 是否使用LLM润色
            calorie_tolerance: 热量容差
            macro_ratios: 宏量营养素比例
            meal_ratios: 三餐热量比例

        Returns:
            完整周食谱 {
                'user_info': {...},
                'tdee': int,
                'daily_recipes': [...],
                'weekly_summary': {...}
            }
        """
        # 1. 计算TDEE
        tdee = self.tdee_estimator.calculate_tdee(user_profile)
        macros = self.tdee_estimator.calculate_macros(tdee)

        # 2. 配置规则引擎
        recipe_config = RecipeConfig(
            tdee=tdee,
            calorie_tolerance=calorie_tolerance,
            macro_ratios=macro_ratios or {"carb": 0.50, "protein": 0.25, "fat": 0.25},
            meal_ratios=meal_ratios or {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
        )

        # 3. 规则引擎生成骨架
        engine = RecipeRuleEngine(self.food_db, recipe_config)
        skeleton = engine.generate_weekly_recipe(
            recent_intake=recent_intake,
            excluded_foods=excluded_foods,
        )

        # 4. LLM润色
        if use_llm_polish:
            try:
                polished = self.llm_polisher.polish_weekly_recipe(skeleton)
            except Exception as e:
                print(f"LLM润色失败，使用原始骨架: {e}")
                polished = skeleton
        else:
            polished = skeleton

        # 5. 生成周汇总
        weekly_summary = self._compute_weekly_summary(polished, tdee)

        # 6. 组装结果
        result = {
            "user_info": {
                "gender": "男" if user_profile.gender.value == "male" else "女",
                "height": f"{user_profile.height_cm}cm",
                "weight": f"{user_profile.weight_kg}kg",
                "age": f"{user_profile.age}岁",
            },
            "tdee": tdee,
            "macros": macros,
            "daily_recipes": polished,
            "weekly_summary": weekly_summary,
        }

        return result

    def _compute_weekly_summary(
        self,
        daily_recipes: List[Dict],
        tdee: float,
    ) -> Dict[str, Any]:
        """计算周汇总统计

        Returns:
            {avg_daily_cal, avg_daily_protein, ..., deviation_days}
        """
        total_cal = []
        total_protein = []
        total_carb = []
        total_fat = []

        for day in daily_recipes:
            day_cal = sum(m["nutrition"].get("calories", 0) for m in day["meals"])
            day_protein = sum(m["nutrition"].get("protein", 0) for m in day["meals"])
            day_carb = sum(m["nutrition"].get("carb", 0) for m in day["meals"])
            day_fat = sum(m["nutrition"].get("fat", 0) for m in day["meals"])

            total_cal.append(day_cal)
            total_protein.append(day_protein)
            total_carb.append(day_carb)
            total_fat.append(day_fat)

        n_days = len(total_cal) or 1
        deviation_count = sum(1 for c in total_cal if abs(c - tdee) > 200)

        return {
            "avg_daily_calories": round(sum(total_cal) / n_days),
            "avg_daily_protein_g": round(sum(total_protein) / n_days),
            "avg_daily_carb_g": round(sum(total_carb) / n_days),
            "avg_daily_fat_g": round(sum(total_fat) / n_days),
            "deviation_days": deviation_count,
            "total_days": len(total_cal),
        }


if __name__ == "__main__":
    planner = WeeklyPlanner(llm_backend="local")

    # 创建用户档案
    profile = UserProfile(
        height_cm=175, weight_kg=70, age=28,
        gender=__import__('recipe.tdee_estimator', fromlist=['Gender']).Gender.MALE,
        activity_level=__import__('recipe.tdee_estimator', fromlist=['ActivityLevel']).ActivityLevel.MODERATE,
    )

    # 生成周食谱
    result = planner.generate(
        user_profile=profile,
        use_llm_polish=False,  # 使用模板润色
    )

    print(f"TDEE: {result['tdee']} kcal")
    print(f"周平均热量: {result['weekly_summary']['avg_daily_calories']} kcal")
    print(f"偏差天数: {result['weekly_summary']['deviation_days']}/{result['weekly_summary']['total_days']}")
