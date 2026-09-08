"""编排器。保持 gradio_demo.py 旧接口兼容。

P0 修正:
- 移除 ±5%/±15% deviation_pct 声明
- 暂停 cooked→raw 自动换算（缺乏可靠依据）
- 未知食物明确标注 unknown_foods，不用同大类冒充
- 不再生成"减重处方"，改为"饮食建议"措辞
"""
from datetime import date, timedelta
from pathlib import Path

from .tdee_estimator import TDEEEstimator, UserProfile
from .nutrition_db import NutritionDB
from .ds_client import DSClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATHS = [
    ROOT / "data/nutrition5k/dishes_verified.csv",
]


class WeeklyPlanner:
    def __init__(self, llm_backend: str = "ds",
                 db_paths=None):
        self.db = NutritionDB(db_paths or DEFAULT_DB_PATHS)
        self.llm = DSClient() if llm_backend in ("ds", "local") else None

    def generate(self, user_profile: UserProfile,
                 use_llm_polish: bool = False,
                 calorie_tolerance: int = 200,
                 start_date=None,
                 enable_substitutions: bool = True,
                 substitutions_per_food: int = 1) -> dict:
        """生成 7 天饮食建议（非医疗处方）。

        P0 修正：
          - 措辞改为"饮食建议"，不输出减重处方
          - 未知食物明确标注 unknown，不冒充
          - 移除 deviation_pct 和 conversion_note
          - 生熟状态保留 DS 标注，不做自动换算
        """
        # 特殊人群门禁：delivery_scope.md L33 末句
        if user_profile.special_populations:
            return {
                "status": "refused",
                "reason": "special_population_route_to_human",
                "audit": {"route": "nutritionist_review"},
            }

        tdee_report = TDEEEstimator(user_profile).estimate()
        target = tdee_report["target_calories"]
        macros = tdee_report["macros_g"]

        profile_summary = (
            f"{user_profile.height_cm}cm/{user_profile.weight_kg}kg/"
            f"{user_profile.age}y/{user_profile.gender.value}"
        )
        try:
            draft = self.llm.plan_draft(profile_summary, int(target),
                                        macros, user_profile.allergies)
        except Exception as e:
            return {
                "status": "llm_failed",
                "reason": str(e),
                "tdee_report": tdee_report,
                "daily_recipes": [],
                "unknown_foods": [],
                "audit_trail": [],
            }

        daily_recipes = []
        unknown_foods = []
        audit_rows = []

        for d in draft.get("daily_plans", []):
            day_rec = {
                "day": d["day"],
                "date": (start_date + timedelta(days=d["day"] - 1)).isoformat() if start_date else None,
                "meals": [],
                "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
            }
            for m in d["meals"]:
                meal_rec = {"meal_name": m["meal"], "foods": [], "nutrition": {}}
                meal_kcal = 0.0
                for item in m["items"]:
                    item_state = item.get("cooked_or_raw", "cooked")
                    # P0: 用 search_with_status 明确状态
                    result = self.db.search_with_status(
                        item["name"], top_k=1,
                        query_cooked_or_raw=item_state,
                    )

                    if result["status"] != "matched" or not result["hits"]:
                        # 未知食物明确标注，不冒充
                        unknown_foods.append({
                            "day": d["day"], "meal": m["meal"],
                            "name": item["name"], "grams": item["grams"],
                            "cooked_or_raw": item_state,
                            "reason": result.get("message", "未知食物"),
                        })
                        meal_rec["foods"].append({
                            "name": item["name"], "grams": item["grams"],
                            "cooked_or_raw": item_state,
                            "nutrition": None,
                            "source": None,
                            "status": "unknown",
                            "message": result.get("message", "未知食物，请手动标注"),
                            "substitutions": [],
                        })
                        continue

                    h = result["hits"][0]
                    factor = item["grams"] / 100.0
                    nut = {
                        "calories": round(h.kcal_per_100g * factor, 1),
                        "protein": round(h.protein_per_100g * factor, 1),
                        "carb": round(h.carb_per_100g * factor, 1),
                        "fat": round(h.fat_per_100g * factor, 1),
                    }
                    meal_kcal += nut["calories"]

                    # P0: 不再生成 deviation_pct 或 conversion_note
                    # 生熟状态保留 DS 标注 + 库标注，让用户判断

                    # 替换选项（保留功能，但 DS 失败时降级）
                    subs = []
                    if enable_substitutions and self.llm:
                        try:
                            sub_resp = self.llm.suggest_substitutions(
                                food_name=item["name"],
                                category="",
                                allergies=user_profile.allergies,
                                target_kcal_per_100g=h.kcal_per_100g,
                                count=substitutions_per_food,
                            )
                            for s in sub_resp.get("substitutions", []):
                                s_result = self.db.search_with_status(
                                    s.get("name", ""), top_k=1,
                                )
                                if s_result["status"] == "matched" and s_result["hits"]:
                                    sh = s_result["hits"][0]
                                    s_factor = (s.get("grams_swap", 100)) / 100.0
                                    subs.append({
                                        "name": s.get("name"),
                                        "grams_swap": s.get("grams_swap"),
                                        "reason": s.get("reason", ""),
                                        "kcal_per_100g_rag": round(sh.kcal_per_100g, 1),
                                        "source": {
                                            "row_id": sh.row_id,
                                            "file": sh.source_file,
                                            "sha256": sh.source_sha256,
                                        },
                                    })
                                else:
                                    # RAG 未命中，标记 unverified
                                    subs.append({
                                        "name": s.get("name"),
                                        "grams_swap": s.get("grams_swap"),
                                        "reason": s.get("reason", ""),
                                        "unverified": True,
                                        "message": "营养数据未在库中核实，请人工确认",
                                    })
                        except Exception:
                            subs = []

                    meal_rec["foods"].append({
                        "name": item["name"], "grams": item["grams"],
                        "cooked_or_raw": item_state,
                        "db_state": h.cooked_or_raw,
                        "nutrition": nut,
                        "source": {
                            "row_id": h.row_id,
                            "file": h.source_file,
                            "sha256": h.source_sha256,
                        },
                        "status": "matched",
                        "substitutions": subs,
                    })
                    audit_rows.append({
                        "day": d["day"], "meal": m["meal"],
                        "food": item["name"], "grams": item["grams"],
                        "row_id": h.row_id, "sha256": h.source_sha256,
                    })
                meal_rec["nutrition"]["calories"] = round(meal_kcal, 1)
                day_rec["meals"].append(meal_rec)
            day_rec["day_total_kcal"] = round(
                sum(m["nutrition"]["calories"] for m in day_rec["meals"]
                    if m["nutrition"]), 1
            )
            day_rec["target_kcal"] = int(target)
            day_rec["deviation_kcal"] = round(day_rec["day_total_kcal"] - target, 1) if day_rec["day_total_kcal"] else None
            day_rec["within_tolerance"] = (
                day_rec["deviation_kcal"] is not None and
                abs(day_rec["deviation_kcal"]) <= calorie_tolerance
            )
            daily_recipes.append(day_rec)

        return {
            "daily_recipes": daily_recipes,
            "unknown_foods": unknown_foods,
            "audit_trail": audit_rows,
            "targets": {"kcal": int(target), "macros_g": macros},
            "tdee_report": tdee_report,
            "disclaimer": "本建议为饮食参考，非医疗处方；具体方案请咨询营养师",
            "unknown_count": len(unknown_foods),
        }
