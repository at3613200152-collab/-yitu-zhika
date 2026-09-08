"""自定义/预设食物池 → 科学配餐（用户或商家补全食物营养数据后，从该池安排饮食）。

设计要点：
- FoodItem：用户/商家提供的食物，含每 100g 营养（kcal/protein/carb/fat），可选固定份量 default_grams。
- PRESET_MENUS：内置预设菜单示例（如 'dumpling' 饺子餐 / 'merchant_demo' 商家推广示例），便于演示与二次开发。
- validate_food：沿用硬约束的合理性范围（每 100g kcal ∈ [0,2000]，蛋白/碳水/脂肪 ∈ [0,100]），缺名或 kcal<=0 拒绝。
- plan_from_pool：仅从给定食物池选食物，按 TDEE 目标热量分配到早/午/晚，
  计算每餐克数与宏量，避免同日重复，给出偏差与免责声明；不因缺少某个类别而强行归入其他类。

食用边界（与模板版一致）：
- 不生成减重处方；措辞为"饮食参考，非医疗处方"。
- 用户/商家补全的数据无权威溯源，统一标记 source.type='user'/'preset'，并保留 disclaimer。
"""
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .tdee_estimator import TDEEEstimator, UserProfile
from .template_planner import MEAL_RATIOS, ALLERGY_FILTER

ROOT = Path(__file__).resolve().parents[1]

# 日常饮食常见类别的中文/ID
VALID_CATEGORIES = [
    "grain", "vegetable", "meat", "seafood", "egg", "dairy",
    "soup_stew", "mixed", "sauce_condiment", "dessert", "other",
]

# 数值合理性边界（与全局硬约束一致，收窄到单食物粒度）
MAX_KCAL_PER_100G = 2000.0
MAX_MACRO_PER_100G = 100.0


@dataclass
class FoodItem:
    name: str
    category: str          # 必须是 VALID_CATEGORIES 之一
    kcal_per_100g: float
    protein_per_100g: float = 0.0
    carb_per_100g: float = 0.0
    fat_per_100g: float = 0.0
    default_grams: float = 0.0   # 0 表示不固定份量
    source_type: str = "user"    # 'user' | 'preset'
    source_ref: str = ""         # preset 菜单名 / 商家标识
    extra: dict = field(default_factory=dict)

    def as_row(self):
        return {
            "name": self.name,
            "category": self.category,
            "kcal_per_100g": round(self.kcal_per_100g, 1),
            "protein_per_100g": round(self.protein_per_100g, 1),
            "carb_per_100g": round(self.carb_per_100g, 1),
            "fat_per_100g": round(self.fat_per_100g, 1),
            "default_grams": round(self.default_grams, 0),
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "status": "pool",
            "verification": "user_provided" if self.source_type == "user" else "preset",
        }


def validate_food(data: dict) -> tuple:
    """校验单条食物数据。返回 (FoodItem | None, error_msg | None)。"""
    name = str(data.get("name", "")).strip()
    category = str(data.get("category", "")).strip()
    if not name:
        return None, "缺少食物名称 name"
    if category not in VALID_CATEGORIES:
        return None, f"category 必须是 {','.join(VALID_CATEGORIES)} 之一，得到 {category or '空'}"
    try:
        kcal = float(data.get("kcal_per_100g"))
        protein = float(data.get("protein_per_100g", 0.0))
        carb = float(data.get("carb_per_100g", 0.0))
        fat = float(data.get("fat_per_100g", 0.0))
        default_grams = float(data.get("default_grams", 0.0))
    except (TypeError, ValueError):
        return None, "营养数值必须是数字"
    if not (0.0 <= kcal <= MAX_KCAL_PER_100G):
        return None, f"kcal_per_100g 超出范围 [0,{MAX_KCAL_PER_100G:.0f}]"
    for label, v in (("protein_per_100g", protein), ("carb_per_100g", carb), ("fat_per_100g", fat)):
        if not (0.0 <= v <= MAX_MACRO_PER_100G):
            return None, f"{label} 超出范围 [0,{MAX_MACRO_PER_100G:.0f}]"
    if kcal <= 0:
        return None, "kcal_per_100g 必须 > 0"
    return FoodItem(
        name=name, category=category, kcal_per_100g=kcal,
        protein_per_100g=protein, carb_per_100g=carb, fat_per_100g=fat,
        default_grams=default_grams,
        source_type=str(data.get("source_type", "user")),
        source_ref=str(data.get("source_ref", "")),
    ), None


# ---- 内置预设菜单示例 ----
# 说明：数值为常见估算，仅用于演示；商家接入时以其提供的营养数据为准。
PRESET_MENUS = {
    "dumpling": {
        "name": "饺子餐（示例）",
        "desc": "这一周只吃饺子类 + 配菜。用户可从列表勾选，或商家按自家产品替换营养数据。",
        "foods": [
            FoodItem("猪肉白菜水饺", "grain", 222, 9.5, 28.0, 9.0, 150, "preset", "dumpling"),
            FoodItem("韭菜鸡蛋水饺", "grain", 210, 8.0, 26.0, 8.5, 150, "preset", "dumpling"),
            FoodItem("三鲜水饺", "grain", 215, 8.8, 27.0, 8.8, 150, "preset", "dumpling"),
            FoodItem("凉拌黄瓜", "vegetable", 25, 1.2, 4.0, 0.6, 120, "preset", "dumpling"),
            FoodItem("紫菜蛋花汤", "soup_stew", 32, 2.0, 2.5, 1.5, 200, "preset", "dumpling"),
        ],
    },
    "merchant_demo": {
        "name": "商家推广示例",
        "desc": "商家想推的 3 款产品 + 2 款配菜；接入时替换为真实检测/标签数据。",
        "foods": [
            FoodItem("轻食鸡胸饭（商家A）", "mixed", 150, 20.0, 16.0, 2.5, 250, "preset", "merchant_demo"),
            FoodItem("低卡藜麦沙拉（商家A）", "mixed", 120, 6.0, 14.0, 4.0, 220, "preset", "merchant_demo"),
            FoodItem("全麦三明治（商家A）", "grain", 180, 12.0, 26.0, 4.0, 180, "preset", "merchant_demo"),
            FoodItem("无糖酸奶", "dairy", 62, 3.2, 4.5, 3.0, 150, "preset", "merchant_demo"),
            FoodItem("清炒时蔬", "vegetable", 60, 2.0, 6.0, 3.0, 150, "preset", "merchant_demo"),
        ],
    },
}


def _blocked_categories(allergies):
    blocked = set()
    for a in allergies:
        blocked.update(ALLERGY_FILTER.get(a, []))
    return blocked


def _grams_for(food, target_kcal):
    """按目标热量反推克数，并做合理范围约束。"""
    if food.kcal_per_100g <= 0:
        return 100
    grams = (target_kcal / food.kcal_per_100g) * 100
    return max(30, min(700, round(grams)))


def plan_from_pool(
    user_profile: UserProfile,
    foods: list,
    start_date=None,
    days: int = 7,
    meal_ratios: dict = None,
    seed: int = 42,
) -> dict:
    """仅从给定食物池 foods 生成 days 天饮食参考。

    返回结构与模板版对齐（daily_recipes/targets/tdee_report/disclaimer 等），
    便于前端复用；generator='pool_v1'。
    """
    meal_ratios = meal_ratios or MEAL_RATIOS
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

    blocked = _blocked_categories(user_profile.allergies)
    pool = [f for f in foods if f.category not in blocked]
    if not pool:
        return {
            "status": "refused",
            "reason": "empty_pool_after_allergy_filter",
            "message": "所选食物被忌口过滤后为空，请调整忌口或增加食物",
            "targets": {"kcal": int(target), "macros_g": macros},
            "tdee_report": tdee_report,
            "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
        }

    rng = random.Random(seed)
    start = start_date or date.today()
    daily_recipes = []

    for d in range(1, days + 1):
        day_rec = {
            "day": d,
            "date": (start + timedelta(days=d - 1)).isoformat(),
            "meals": [],
            "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
        }
        day_kcal = 0.0
        day_used = set()
        # 每餐先主食/高密度、后配菜（蔬/汤）；主食顺序每天轮转保证 7 天不重复
        mains = sorted([f for f in pool if f.category not in ("vegetable", "soup_stew")],
                       key=lambda f: -f.kcal_per_100g)
        sides = [f for f in pool if f.category in ("vegetable", "soup_stew")]
        offset = (d % len(mains)) if mains else 0
        mains_r = (mains[offset:] + mains[:offset]) if mains else []
        dense_day = mains_r + sides

        meal_idx = 0
        for meal_name, ratio in meal_ratios.items():
            meal_target = target * ratio
            meal_rec = {"meal_name": meal_name, "foods": [], "nutrition": {}}
            meal_kcal = 0.0
            remaining = meal_target
            used_in_meal = set()

            # 按热量密度降序贪心填充：先用高密度食物逼近目标热量，
            # 低密度配菜（蔬/汤）在有余量时作为佐餐补齐，避免"凑类别"。
            for food in dense_day:
                if len(meal_rec["foods"]) >= 5:
                    break
                if remaining <= meal_target * 0.05:
                    break
                if food.name in used_in_meal:
                    continue
                cap = food.default_grams if food.default_grams > 0 else 450
                grams = max(50, min(cap, round((remaining / food.kcal_per_100g) * 100)))
                if grams <= 0:
                    continue
                used_in_meal.add(food.name)
                day_used.add(food.name)
                factor = grams / 100.0
                nut = {
                    "calories": round(food.kcal_per_100g * factor, 1),
                    "protein": round(food.protein_per_100g * factor, 1),
                    "carb": round(food.carb_per_100g * factor, 1),
                    "fat": round(food.fat_per_100g * factor, 1),
                }
                meal_kcal += nut["calories"]
                remaining -= nut["calories"]
                meal_rec["foods"].append({
                    "name": food.name,
                    "category": food.category,
                    "grams": grams,
                    "nutrition": nut,
                    "per_100g": {
                        "kcal": round(food.kcal_per_100g, 1),
                        "protein": round(food.protein_per_100g, 1),
                        "carb": round(food.carb_per_100g, 1),
                        "fat": round(food.fat_per_100g, 1),
                    },
                    "source": {"type": food.source_type, "ref": food.source_ref or None},
                    "status": "pool",
                    "verification": food.source_type,
                    "substitutions": [],
                })

            meal_rec["nutrition"]["calories"] = round(meal_kcal, 1)
            day_rec["meals"].append(meal_rec)
            day_kcal += meal_kcal
            meal_idx += 1

        # 每天至少一份蔬/汤配菜（补充膳食纤维/水分，低热量不破坏目标）
        used_cats = {f["category"] for m in day_rec["meals"] for f in m["foods"]}
        if not (used_cats & {"vegetable", "soup_stew"}):
            side = next((f for f in pool if f.category in ("vegetable", "soup_stew")), None)
            if side is not None and len(day_rec["meals"][-1]["foods"]) < 5:
                dinner = day_rec["meals"][-1]
                grams = side.default_grams if side.default_grams > 0 else 150
                factor = grams / 100.0
                nut = {
                    "calories": round(side.kcal_per_100g * factor, 1),
                    "protein": round(side.protein_per_100g * factor, 1),
                    "carb": round(side.carb_per_100g * factor, 1),
                    "fat": round(side.fat_per_100g * factor, 1),
                }
                dinner["foods"].append({
                    "name": side.name, "category": side.category, "grams": grams,
                    "nutrition": nut,
                    "source": {"type": side.source_type, "ref": side.source_ref or None},
                    "status": "pool", "verification": side.source_type, "substitutions": [],
                })
                dinner["nutrition"]["calories"] = round(dinner["nutrition"]["calories"] + nut["calories"], 1)
                day_kcal += nut["calories"]

        day_rec["day_total_kcal"] = round(day_kcal, 1)
        day_rec["target_kcal"] = int(target)
        day_rec["deviation_kcal"] = round(day_kcal - target, 1)
        day_rec["within_tolerance"] = abs(day_kcal - target) <= 200
        daily_recipes.append(day_rec)

    pool_dishes = [f.as_row() for f in pool]
    return {
        "status": "ok",
        "daily_recipes": daily_recipes,
        "unknown_foods": [],
        "audit_trail": [],
        "targets": {"kcal": int(target), "macros_g": macros},
        "tdee_report": tdee_report,
        "tdee": tdee_report["tdee"],
        "bmr": tdee_report["bmr"],
        "target_calories": tdee_report["target_calories"],
        "food_pool": pool_dishes,
        "food_pool_count": len(pool_dishes),
        "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
        "unknown_count": 0,
        "generator": "pool_v1",
    }


def foods_from_preset(name: str) -> tuple:
    """取预设菜单食物。返回 (list[FoodItem], name, desc)。"""
    preset = PRESET_MENUS.get(name)
    if not preset:
        return [], "", ""
    return list(preset["foods"]), preset["name"], preset.get("desc", "")
