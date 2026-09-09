"""P1-A：标签规则审计 —— 量化旧子串匹配规则的误分类规模。

对比 旧规则(裸子串+类别优先级) vs 新规则(词边界+右端/最长短语优先+人工覆盖)。
输出：
  - 每个食材：old_cat / new_cat / matched_keyword / 影响餐盘数
  - 菜品级：final category 发生变化(40%阈值 mixed 规则)的数量
  - 写入 artifacts/category-label-audit-<stamp>/evidence.json
用法：python scripts/audit_category_labels.py [--data_dir data/Nutrition5k] [--output artifacts/...]
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_categories import classify_ingredient, classify_ingredient_legacy, match_ingredient


def load_rows(data_dir):
    p = Path(data_dir) / "dish_ingredients.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dish = (r.get("dish_id") or "").strip()
            ingr = (r.get("ingr_name") or "").strip()
            if not dish or not ingr:
                continue
            try:
                grams = float(r.get("grams") or 0)
            except (TypeError, ValueError):
                grams = 0.0
            if grams <= 0:
                continue
            rows.append({"dish": dish, "ingr": ingr, "grams": grams})
    return rows


def dish_labels(rows, classify_fn):
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        cat = classify_fn(r["ingr"])
        agg[r["dish"]][cat] += r["grams"]
    out = {}
    for dish, weights in agg.items():
        total = sum(weights.values())
        best = max(weights, key=weights.get)
        out[dish] = best if (best and weights[best] / total >= 0.40) else "mixed"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/Nutrition5k")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = load_rows(args.data_dir)
    ingr_counts = {}
    for r in rows:
        ingr_counts[r["ingr"]] = ingr_counts.get(r["ingr"], 0) + 1  # dish mentions (approx)

    # 食材级：每种食材 old vs new，统计影响的餐盘数（此处以 mention 数近似，非独立餐盘去重）
    ingr_table = {}
    for ingr in sorted(ingr_counts):
        new_cat, kw = match_ingredient(ingr)
        old_cat = classify_ingredient_legacy(ingr)
        ingr_table[ingr] = {
            "old_cat": old_cat, "new_cat": new_cat, "matched_keyword": kw,
            "mentions": ingr_counts[ingr], "changed": old_cat != new_cat,
        }
    changed_ingr = [k for k, v in ingr_table.items() if v["changed"]]

    # 菜品级：final label 变化
    old_labels = dish_labels(rows, classify_ingredient_legacy)
    new_labels = dish_labels(rows, classify_ingredient)
    changed_dishes = [d for d in old_labels if old_labels[d] != new_labels.get(d)]
    dish_changes = {d: {"old": old_labels[d], "new": new_labels.get(d)} for d in changed_dishes}

    summary = {
        "n_rows": len(rows),
        "n_unique_ingredients": len(ingr_table),
        "n_ingredients_changed_rule": len(changed_ingr),
        "n_dishes": len(old_labels),
        "n_dishes_label_changed": len(changed_dishes),
        "changed_ingredients": changed_ingr[:200],
        "changed_dishes": dish_changes,
    }

    out_path = args.output or None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({"summary": summary, "ingredient_table": ingr_table,
                                              "dish_label_changes": dish_changes}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"证据写入: {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 展示若干典型 changed 食材
    print("\n典型误匹配（old→new）, 示例:")
    for ingr in list(changed_ingr)[:25]:
        v = ingr_table[ingr]
        print(f"  {ingr:32s} {v['old_cat']:16s}-> {v['new_cat']:16s} (kw={v['matched_keyword']}, mentions={v['mentions']})")


if __name__ == "__main__":
    main()
