"""label_schema v2：12 类（新增 fruit）+ 修正后的五目标 manifest。

- 基础：results/meal_macros_expanded_v1/manifest.json 的 3390 行。划分 / 图像 / 五目标 / mask
  **一律不动**，只重算 category 与 category_idx，保证与旧模型的差异仅来自标签。
- 规则：当前 src/data/build_categories.py（词边界 + 右端最长 + OVERRIDES + fruit 类），
  从 data/Nutrition5k/dish_ingredients.csv 按质量加权（最高类 >=40%，否则 mixed）重算。
- 12 类 category_to_idx（字母序，含 fruit）。
- 溯源：规则代码 SHA、食材表 SHA、基础 manifest SHA、逐 split 类别计数、变更明细。
- 输出 results/meal_macros_corrected_v2/{manifest.json,label_changes.json}；**不覆盖任何旧清单**。

用法：python src/data/meal_macros_corrected_manifest.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest
from src.data.build_categories import build_dish_categories

BASE_MANIFEST = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
OUT_DIR = ROOT / 'results/meal_macros_corrected_v2'
DATA_DIR = ROOT / 'data/Nutrition5k'
LABEL_SCHEMA_VERSION = 'v2_fruit_12class'
RULE_CODE = ROOT / 'src/data/build_categories.py'
INGREDIENTS_CSV = DATA_DIR / 'dish_ingredients.csv'
# 12 类：在旧 11 类基础上加入 fruit（字母序）
CATEGORY_ORDER = ['dairy', 'dessert', 'egg', 'fruit', 'grain', 'meat', 'mixed',
                  'other', 'sauce_condiment', 'seafood', 'soup_stew', 'vegetable']


def build():
    base = json.loads(BASE_MANIFEST.read_text(encoding='utf-8'))
    dish_categories = build_dish_categories(str(DATA_DIR))
    category_to_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}

    changes, missing, unknown_cat = {}, [], set()
    rows = []
    for row in base['rows']:
        r = dict(row)
        old = r['category']
        new = dish_categories.get(r['dish_id'])
        if new is None:
            missing.append(r['dish_id'])
            new = old  # 无食材记录：保留旧标签，并在报告中列出
        if new not in category_to_idx:
            unknown_cat.add(new)
        r['category'] = new
        r['category_idx'] = category_to_idx[new]
        rows.append(r)
        if new != old:
            changes[r['dish_id']] = {'split': r['split'], 'old': old, 'new': new}

    if unknown_cat:
        raise ValueError(f'重算出现未知类别（不在 CATEGORY_ORDER 内）: {sorted(unknown_cat)}')

    counts = Counter(r['split'] for r in rows)
    transitions = Counter(f"{c['old']}->{c['new']}" for c in changes.values())
    manifest = {
        'protocol': 'nutrition5k_overhead_official_ids_macros_expanded_corrected_v2',
        'frozen_protocol': base['frozen_protocol'],
        'derived_from': str(BASE_MANIFEST.relative_to(ROOT)),
        'derived_from_sha256': file_digest(BASE_MANIFEST),
        'label_schema_version': LABEL_SCHEMA_VERSION,
        'label_rule_code_sha256': file_digest(RULE_CODE),
        'label_rule_note': '词边界 + 右端/最长短语 + OVERRIDES；新增 fruit 类；salad 复合名按主料归类',
        'ingredients_sha256': file_digest(INGREDIENTS_CSV),
        'target_names': base['target_names'], 'target_units': base['target_units'],
        'macro_names': base['macro_names'],
        'category_to_idx': category_to_idx,
        'category_source': 'derived_ingredient_mass_keyword_groups_not_official_food_classes',
        'target_stats': base['target_stats'],
        'counts': dict(counts),
        'category_counts': {s: dict(Counter(r['category'] for r in rows if r['split'] == s)) for s in counts},
        'valid_per_field': base['valid_per_field'],
        'metadata_sha256': base['metadata_sha256'],
        'new_dishes_from_expanded': base.get('new_dishes_from_expanded'),
        'label_changes_vs_base': {
            'n_changed': len(changes),
            'n_rows': len(rows),
            'by_transition': dict(transitions.most_common()),
            'detail_file': 'label_changes.json',
        },
        'rows_missing_ingredient_records': missing,
        'missing_handling': base['missing_handling'],
        'limitations': list(base['limitations']) + [
            'label_schema v2 为关键词+质量阈值推导，非官方食物类别；fruit 采用烹饪习惯口径'
            '（番茄/鳄梨/橄榄/甜椒仍属 vegetable）。',
            '番茄酱类/豆类/坚果等仍无专用类别，落 other；未做官方食物类别对齐。',
        ],
        'rows': rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT_DIR / 'manifest.json', manifest)
    atomic_json(OUT_DIR / 'label_changes.json', {
        'protocol': manifest['protocol'], 'label_schema_version': LABEL_SCHEMA_VERSION,
        'base': str(BASE_MANIFEST.relative_to(ROOT)), 'base_sha256': manifest['derived_from_sha256'],
        'n_changed': len(changes), 'by_transition': dict(transitions.most_common()),
        'changes': changes})

    print(json.dumps({
        'manifest': str((OUT_DIR / 'manifest.json').relative_to(ROOT)),
        'sha256': file_digest(OUT_DIR / 'manifest.json'),
        'rows': len(rows), 'counts': dict(counts),
        'categories': len(category_to_idx),
        'label_changes': len(changes), 'by_transition': dict(transitions.most_common()),
        'category_counts_train': manifest['category_counts']['train'],
        'missing_ingredient_rows': len(missing),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    build()
