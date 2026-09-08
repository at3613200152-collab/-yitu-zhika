"""P1.2 下周食谱首版 - 模板化生成（不依赖 DS API）

核心理念:
- 使用 NutritionDB 中已核验的食物数据
- 按 category 选食物，保证营养数据可溯源
- 忌口过滤（allergies 列表）
- 替换选项：用同类别的另一个核验食物
- 不生成"减重处方"，明确标注为"饮食参考"
- 7 天 × 3 餐模板，按 TDEE 目标热量分配
"""
from datetime import date, timedelta
from pathlib import Path
import random

from .tdee_estimator import TDEEEstimator, UserProfile
from .nutrition_db import NutritionDB

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATHS = [
    ROOT / "data/nutrition5k/dishes_verified.csv",
]

# 餐次热量分配比例（早/午/晚）
MEAL_RATIOS = {"breakfast": 0.25, "lunch": 0.40, "dinner": 0.35}

# 每餐按 category 分配食物数量
MEAL_STRUCTURE = {
    "breakfast": ["grain", "dairy"],      # 主食 + 乳/蛋
    "lunch": ["grain", "vegetable", "meat"],  # 主食 + 蔬 + 肉
    "dinner": ["vegetable", "meat", "grain"],  # 蔬 + 肉 + 主食
}

# 忌口映射：用户输入 → category 过滤
ALLERGY_FILTER = {
    "鸡蛋": ["egg"],
    "牛奶": ["dairy"],
    "海鲜": ["seafood"],
    "花生": ["other"],   # other 可能含坚果
    "坚果": ["other"],
    "牛肉": ["meat"],
    "猪肉": ["meat"],
    "羊肉": ["meat"],
}


class TemplatePlanner:
    """模板化食谱生成器，不依赖 DS API。"""

    def __init__(self, db_paths=None):
        self.db = NutritionDB(db_paths or DEFAULT_DB_PATHS)
        # 按 category 分组所有食物
        self.by_category = {}
        for row in self.db.rows:
            cat = row.get("category", "")
            if cat not in self.by_category:
                self.by_category[cat] = []
            self.by_category[cat].append(row)
        # 确保可复现
        for cat in self.by_category:
            self.by_category[cat].sort(key=lambda r: r["kcal_per_100g"])

    def _filter_allergies(self, categories, allergies):
        """根据忌口过滤 category。"""
        blocked_cats = set()
        for allergy in allergies:
            blocked_cats.update(ALLERGY_FILTER.get(allergy, []))
        return [c for c in categories if c not in blocked_cats]

    def _pick_food(self, category, target_kcal, used_ids):
        """从 category 中选一个食物，匹配目标热量。"""
        candidates = self.by_category.get(category, [])
        if not candidates:
            return None
        # 排除已用
        available = [r for r in candidates if r["row_id"] not in used_ids]
        if not available:
            available = candidates
        # 按热量密度选最接近目标的食物
        # 目标：找一个食物，100g 的热量约等于目标
        best = min(available, key=lambda r: abs(r["kcal_per_100g"] - target_kcal))
        return best

    def _calc_grams(self, food, target_kcal):
        """根据目标热量反推克数。"""
        if food["kcal_per_100g"] <= 0:
            return 100  # 默认 100g
        grams = (target_kcal / food["kcal_per_100g"]) * 100
        # 限制在合理范围
        return max(50, min(500, round(grams)))

    def _get_substitution(self, food, used_ids):
        """从同类别找替换。"""
        cat = food.get("category", "")
        candidates = self.by_category.get(cat, [])
        available = [r for r in candidates if r["row_id"] != food["row_id"]
                     and r["row_id"] not in used_ids]
        if not available:
            return None
        # 选热量最接近的
        return min(available, key=lambda r: abs(r["kcal_per_100g"] - food["kcal_per_100g"]))

    def generate(self, user_profile: UserProfile,
                 start_date=None,
                 days: int = 7) -> dict:
        """生成 7 天模板食谱。

        不依赖 DS API，所有食物都来自 NutritionDB 核验数据。
        不生成"减重处方"，明确标注为"饮食参考"。
        """
        # 特殊人群门禁
        if user_profile.special_population:
            return {
                "status": "refused",
                "reason": "special_population_route_to_human",
                "audit": {"route": "nutritionist_review"},
                "disclaimer": "特殊人群需营养师审核，本工具不生成方案",
            }

        tdee_report = TDEEEstimator(user_profile).estimate()
        target = tdee_report["target_calories"]
        macros = tdee_report["macros_g"]

        used_ids = set()
        daily_recipes = []
        unknown_foods = []

        start = start_date or date.today()

        for d in range(1, days + 1):
            day_rec = {
                "day": d,
                "date": (start + timedelta(days=d - 1)).isoformat(),
                "meals": [],
                "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
            }
            day_kcal = 0.0

            for meal_name, ratio in MEAL_RATIOS.items():
                meal_target = target * ratio
                meal_cats = MEAL_STRUCTURE.get(meal_name, [])
                # 忌口过滤
                meal_cats = self._filter_allergies(meal_cats, user_profile.allergies)

                meal_rec = {"meal_name": meal_name, "foods": [], "nutrition": {}}
                meal_kcal = 0.0

                # 每个 category 平均分热量
                per_cat_target = meal_target / max(len(meal_cats), 1)

                for cat in meal_cats:
                    food = self._pick_food(cat, per_cat_target, used_ids)
                    if not food:
                        unknown_foods.append({
                            "day": d, "meal": meal_name,
                            "name": cat, "category": cat,
                            "reason": "无可用食物",
                        })
                        continue

                    used_ids.add(food["row_id"])
                    grams = self._calc_grams(food, per_cat_target)
                    factor = grams / 100.0

                    nut = {
                        "calories": round(food["kcal_per_100g"] * factor, 1),
                        "protein": round(food["protein_per_100g"] * factor, 1),
                        "carb": round(food["carb_per_100g"] * factor, 1),
                        "fat": round(food["fat_per_100g"] * factor, 1),
                    }
                    meal_kcal += nut["calories"]

                    # 生成替换选项（同类别）
                    sub_food = self._get_substitution(food, used_ids)
                    sub_list = []
                    if sub_food:
                        sub_grams = self._calc_grams(sub_food, per_cat_target)
                        sub_list.append({
                            "name": sub_food["name"],
                            "grams_swap": sub_grams,
                            "reason": f"同类替换（{cat}）",
                            "kcal_per_100g_rag": round(sub_food["kcal_per_100g"], 1),
                            "source": {
                                "row_id": sub_food["row_id"],
                                "file": sub_food["source_file"],
                                "sha256": sub_food["source_sha256"],
                            },
                        })

                    meal_rec["foods"].append({
                        "name": food["name"],
                        "category": cat,
                        "grams": grams,
                        "cooked_or_raw": food.get("cooked_or_raw", "cooked"),
                        "nutrition": nut,
                        "per_100g": {
                            "kcal": round(food["kcal_per_100g"], 1),
                            "protein": round(food["protein_per_100g"], 1),
                            "carb": round(food["carb_per_100g"], 1),
                            "fat": round(food["fat_per_100g"], 1),
                        },
                        "source": {
                            "row_id": food["row_id"],
                            "file": food["source_file"],
                            "sha256": food["source_sha256"],
                        },
                        "status": "matched",
                        "substitutions": sub_list,
                    })

                meal_rec["nutrition"]["calories"] = round(meal_kcal, 1)
                day_rec["meals"].append(meal_rec)
                day_kcal += meal_kcal

            day_rec["day_total_kcal"] = round(day_kcal, 1)
            day_rec["target_kcal"] = int(target)
            day_rec["deviation_kcal"] = round(day_kcal - target, 1)
            day_rec["within_tolerance"] = abs(day_kcal - target) <= 200
            daily_recipes.append(day_rec)

        return {
            "status": "ok",
            "daily_recipes": daily_recipes,
            "unknown_foods": unknown_foods,
            "audit_trail": [],
            "targets": {"kcal": int(target), "macros_g": macros},
            "tdee_report": tdee_report,
            # 顶层别名：供任何直接读 tdee 的调用方使用（旧接口曾期望顶层非空）
            "tdee": tdee_report["tdee"],
            "bmr": tdee_report["bmr"],
            "target_calories": tdee_report["target_calories"],
            "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
            "unknown_count": len(unknown_foods),
            "generator": "template_v1",
        }


if __name__ == "__main__":
    # 自测
    import sys
    sys.path.insert(0, str(ROOT))

    db_path = ROOT / "data/nutrition5k/dishes_verified.csv"
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(0)

    planner = TemplatePlanner()
    print(f"Categories: {list(planner.by_category.keys())}")
    for cat, foods in planner.by_category.items():
        print(f"  {cat}: {len(foods)} foods")

    # 简单生成测试
    from .tdee_estimator import UserProfile, Gender, ActivityLevel, Goal
    profile = UserProfile(
        height_cm=170, weight_kg=70, age=30,
        gender=Gender.MALE,
        activity_level=ActivityLevel.MODERATE,
        goal=Goal.MAINTAIN,
        allergies=[],
    )
    result = planner.generate(profile)
    print(f"\nStatus: {result['status']}")
    print(f"Days: {len(result['daily_recipes'])}")
    if result["daily_recipes"]:
        day1 = result["daily_recipes"][0]
        print(f"Day 1 target: {day1['target_kcal']} kcal, actual: {day1['day_total_kcal']} kcal")
        for m in day1["meals"]:
            print(f"  {m['meal_name']}: {m['nutrition']['calories']} kcal")
            for f in m["foods"]:
                print(f"    {f['name']} {f['grams']}g → {f['nutrition']['calories']} kcal")
                for s in f.get("substitutions", []):
                    print(f"      替换: {s['name']} {s['grams_swap']}g")
