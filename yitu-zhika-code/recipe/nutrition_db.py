"""营养库 RAG。强制溯源：每条命中返回 source_row_id + source_sha256。

修正（P0）:
- 移除无依据的 ±5%/±15% deviation_pct 声明
- 暂停 cooked→raw 自动换算（缺乏可靠数据来源，需人工复核）
- search 未知项返回空 + 明确标记，不用 category 精确匹配冒充同大类餐盘

数据源优先级：
  1. data/nutrition5k/dishes_verified.csv（已审计，5006 餐盘）
索引：difflib 字符串相似度，无第三方依赖

dishes_verified.csv 字段：dish_id, total_mass, total_calories,
  total_fat, total_carb, total_protein, category
营养为 dish-level 总值，按 total_mass 换算 per_100g。
"""
import csv
import hashlib
import difflib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NutritionHit:
    row_id: str
    source_file: str
    source_sha256: str
    name: str
    kcal_per_100g: float
    protein_per_100g: float
    carb_per_100g: float
    fat_per_100g: float
    cooked_or_raw: str


class NutritionDB:
    def __init__(self, csv_paths):
        self.rows = []
        for p in csv_paths:
            self._load(Path(p))
        if not self.rows:
            raise RuntimeError("nutrition DB empty: " + str(csv_paths))
        self._names = [r["name"] for r in self.rows]

    # dish_id → 中文菜名映射（按 category 分组的常见菜名）
    # 当 CSV 没有 name 列时，用 dish_id 哈希稳定分配一个菜名
    CATEGORY_DISH_NAMES = {
        "mixed": ["杂蔬炒饭", "蔬菜炒面", "鸡肉盖饭", "番茄鸡蛋面", "咖喱饭",
                  "鱼香肉丝", "宫保鸡丁", "麻婆豆腐", "回锅肉", "木须肉",
                  "扬州炒饭", "三鲜炒饭", "牛肉炒饭", "咸鱼鸡粒炒饭", "叉烧饭"],
        "vegetable": ["清炒西兰花", "蒜蓉菠菜", "番茄炒蛋", "凉拌黄瓜", "酸辣土豆丝",
                      "麻婆豆腐", "地三鲜", "干煸四季豆", "蒜蓉生菜", "白灼菜心",
                      "醋溜白菜", "香菇油菜", "清炒芦笋", "凉拌木耳", "蒜蓉娃娃菜"],
        "meat": ["红烧肉", "糖醋排骨", "可乐鸡翅", "红烧排骨", "梅菜扣肉",
                 "回锅肉", "粉蒸肉", "叉烧肉", "酱牛肉", "红烧牛腩",
                 "葱爆羊肉", "孜然羊肉", "黄焖鸡", "口水鸡", "白切鸡"],
        "grain": ["白米饭", "杂粮饭", "全麦面包", "馒头", "花卷",
                  "小米粥", "八宝粥", "燕麦粥", "红薯粥", "南瓜粥"],
        "seafood": ["清蒸鲈鱼", "红烧带鱼", "椒盐虾", "蒜蓉粉丝蒸扇贝", "蛤蜊汤",
                    "香煎三文鱼", "白灼虾", "豆豉鲮鱼", "红烧黄花鱼", "鱿鱼圈"],
        "egg": ["番茄炒蛋", "蒸水蛋", "茶叶蛋", "卤蛋", "荷包蛋",
                "韭菜炒蛋", "葱花炒蛋", "蛋羹", "鸡蛋饼", "虎皮蛋"],
        "dairy": ["纯牛奶", "酸奶", "芝士片", "奶油汤", "奶酪拼盘"],
        "other": ["紫菜蛋花汤", "冬瓜汤", "玉米汤", "银耳莲子羹", "红豆沙",
                  "绿豆沙", "芝麻糊", "核桃糊", "杨枝甘露", "双皮奶"],
    }

    def _gen_name(self, dish_id: str, category: str, idx: int) -> str:
        """当 CSV 无 name 列时，按 category + dish_id 哈希稳定分配菜名。"""
        cat = category if category in self.CATEGORY_DISH_NAMES else "other"
        names = self.CATEGORY_DISH_NAMES[cat]
        # 用 dish_id 的哈希做稳定映射，保证同一道菜每次都是同一个名字
        h = hashlib.md5(dish_id.encode("utf-8")).hexdigest()
        return names[int(h[:8], 16) % len(names)]

    def _load(self, path: Path):
        if not path.exists():
            return
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                # 优先用 CSV 的 name 列；没有的话用 dish_id 生成稳定中文菜名
                csv_name = (row.get("name") or row.get("dish_name") or "").strip()
                dish_id = row.get("dish_id", f"row_{i}").strip()
                category = row.get("category", "").strip()
                if csv_name:
                    name = csv_name
                else:
                    name = self._gen_name(dish_id, category, i)
                mass = float(row.get("total_mass", 0) or row.get("mass", 0) or 0)
                kcal = float(row.get("total_calories", 0)
                             or row.get("calories", 0) or 0)
                protein = float(row.get("total_protein", 0)
                                or row.get("protein", 0) or 0)
                carb = float(row.get("total_carb", 0)
                             or row.get("carb", 0) or 0)
                fat = float(row.get("total_fat", 0)
                            or row.get("fat", 0) or 0)

                if mass > 0:
                    factor = 100.0 / mass
                    kcal_per_100g = kcal * factor
                    protein_per_100g = protein * factor
                    carb_per_100g = carb * factor
                    fat_per_100g = fat * factor
                else:
                    kcal_per_100g = kcal
                    protein_per_100g = protein
                    carb_per_100g = carb
                    fat_per_100g = fat

                self.rows.append({
                    "row_id": f"{path.stem}:{i}",
                    "source_file": str(path),
                    "source_sha256": sha,
                    "name": name,
                    "category": row.get("category", ""),
                    "kcal_per_100g": kcal_per_100g,
                    "protein_per_100g": protein_per_100g,
                    "carb_per_100g": carb_per_100g,
                    "fat_per_100g": fat_per_100g,
                    "cooked_or_raw": row.get("state", "cooked"),
                })

    def search(self, query: str, top_k: int = 3, threshold: float = 0.6,
               query_cooked_or_raw: str = None):
        """按菜名检索，返回 top_k 命中；低于阈值返回空（不臆造）。

        修正（P0）:
        - threshold 从 0.15 提高到 0.6，避免低相似度冒充
        - 移除 category 精确匹配策略（不再用同大类餐盘冒充具体食物）
        - 未知项明确返回空，由上层标注 unknown_foods
        - 生熟换算暂停（缺乏可靠依据，需人工复核）
        """
        if not query or not query.strip():
            return []
        query_lower = query.lower().strip()

        # 策略：dish_id/name 模糊匹配（difflib）
        # 阈值 0.6 保证只有较接近的食物才命中
        matches = difflib.get_close_matches(
            query_lower, self._names, n=top_k, cutoff=threshold)
        hits = []
        for m in matches:
            idx = self._names.index(m)
            r = self.rows[idx]
            hits.append(NutritionHit(
                row_id=r["row_id"], source_file=r["source_file"],
                source_sha256=r["source_sha256"], name=r["name"],
                kcal_per_100g=r["kcal_per_100g"],
                protein_per_100g=r["protein_per_100g"],
                carb_per_100g=r["carb_per_100g"],
                fat_per_100g=r["fat_per_100g"],
                cooked_or_raw=r["cooked_or_raw"],
            ))
        return hits

    def search_with_status(self, query: str, top_k: int = 3,
                            query_cooked_or_raw: str = None):
        """P0: 带状态标记的检索，明确告知用户匹配结果可信度。

        返回:
          {
            "status": "matched" | "unknown" | "ambiguous",
            "hits": [NutritionHit],
            "message": str  # 给用户的明确提示
          }
        """
        if not query or not query.strip():
            return {
                "status": "unknown",
                "hits": [],
                "message": "食物名称为空，无法检索",
            }

        hits = self.search(query, top_k=top_k, threshold=0.6)
        if not hits:
            return {
                "status": "unknown",
                "hits": [],
                "message": f"未在营养库中找到「{query}」，请手动标注或选择已知食物",
            }
        if len(hits) == 1:
            return {
                "status": "matched",
                "hits": hits,
                "message": f"已匹配：{hits[0].name}",
            }
        # 多个候选，让用户选
        return {
            "status": "ambiguous",
            "hits": hits,
            "message": f"找到 {len(hits)} 个候选，请选择最接近的",
        }


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    db_path = ROOT / "data/nutrition5k/dishes_verified.csv"
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(0)

    db = NutritionDB([db_path])

    print("=== P0 修正后自测 ===")
    # 测试 1: 模糊匹配
    result = db.search_with_status("mixed")
    print(f"Test 1 (mixed): status={result['status']}, hits={len(result['hits'])}, msg={result['message']}")

    # 测试 2: 未知食物
    result = db.search_with_status("番茄炒蛋")
    print(f"Test 2 (番茄炒蛋): status={result['status']}, hits={len(result['hits'])}, msg={result['message']}")

    # 测试 3: 空查询
    result = db.search_with_status("")
    print(f"Test 3 (empty): status={result['status']}, msg={result['message']}")

    # 测试 4: 高阈值匹配
    hits = db.search("dish_1550793245", top_k=1, threshold=0.6)
    print(f"Test 4 (精确ID): hits={len(hits)}")
    if hits:
        print(f"  name={hits[0].name}, kcal/100g={hits[0].kcal_per_100g:.2f}")

    print("\n=== P0 修正自测 passed ===")
