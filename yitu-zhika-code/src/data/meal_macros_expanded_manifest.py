"""P1-C：扩展五目标 manifest（冻结 3262 + 新增 128 餐盘），带逐字段 mask / 单位 / 额外溯源字段。

- 旧盘：复用 results/meal_macros_v1/manifest.json 的 5-target + mask + 溯源（不动划分）。
- 新 128 盘：来自 results/meal_expanded_v2/manifest.json（split/image/views/cooking_method/
  retained_oil_grams/mass_basis/source），宏量从 data/Nutrition5k/dishes_verified.csv 按字段名取。
- 输出 results/meal_macros_expanded_v1/manifest.json；train-only 逐字段统计。
- 用途：GPU 上做 A/B（旧 3262 vs 扩展 3390，同模型/预算）测"补数据收益"。不覆盖冻结清单。
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest

TARGETS = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']
UNITS = ['kcal', 'g', 'g', 'g', 'g']
CSV_FIELDS = ['total_calories', 'total_mass', 'total_protein', 'total_carb', 'total_fat']
MACRO_NAMES = {'protein': 'protein_g', 'carbohydrate': 'carbohydrate_g', 'fat': 'fat_g'}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def macros_from_csv(macro):
    raw = [_f(macro.get(c)) for c in CSV_FIELDS]
    if raw[0] is None or raw[1] is None or raw[1] <= 0:
        return None
    targets = [raw[0], raw[1],
               raw[2] if raw[2] is not None and raw[2] >= 0 else 0.0,
               raw[3] if raw[3] is not None and raw[3] >= 0 else 0.0,
               raw[4] if raw[4] is not None and raw[4] >= 0 else 0.0]
    mask = [1, 1] + [1 if raw[i] is not None and raw[i] >= 0 else 0 for i in (2, 3, 4)]
    return targets, mask


def target_stats(rows):
    out = {}
    for i, name in enumerate(TARGETS):
        vals = [r['targets'][i] for r in rows if r['split'] == 'train' and r['mask'][i] == 1]
        if len(vals) < 2:
            raise ValueError(f'Insufficient train samples for {name}')
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        if not math.isfinite(std) or std <= 0:
            raise ValueError(f'Degenerate train stats for {name}')
        out[name] = {'mean': round(mean, 4), 'std': round(std, 4), 'samples': len(vals), 'unit': UNITS[i],
                     'source': 'train_only'}
    return out


def build():
    macros_manifest = json.loads((ROOT / 'results/meal_macros_v1/manifest.json').read_text(encoding='utf-8'))
    expanded = json.loads((ROOT / 'results/meal_expanded_v2/manifest.json').read_text(encoding='utf-8'))
    csv_path = ROOT / 'data/Nutrition5k/dishes_verified.csv'
    output = ROOT / 'results/meal_macros_expanded_v1/manifest.json'

    macro_by_dish = {}
    with csv_path.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            macro_by_dish[r['dish_id']] = r
    frozen_ids = {r['dish_id'] for r in macros_manifest['rows']}

    old_rows = macros_manifest['rows']
    new_rows = []
    for er in expanded['rows']:
        if er['dish_id'] in frozen_ids:
            continue
        mm = macros_from_csv(macro_by_dish.get(er['dish_id'], {}))
        if mm is None:
            raise ValueError(f'Missing/overhead-invalid macros for {er["dish_id"]}')
        targets, mask = mm
        new_rows.append({
            'dish_id': er['dish_id'], 'split': er['split'], 'capture_day_utc': er.get('capture_day_utc'),
            'image': er['image'], 'image_sha256': er.get('image_sha256'), 'pixel_sha256': er.get('pixel_sha256'),
            'size': er.get('size'), 'targets': targets, 'mask': mask,
            'category': er['category'], 'category_idx': er['category_idx'],
            'views': er.get('views'), 'source': er.get('source'),
            'cooking_method': er.get('cooking_method'), 'retained_oil_grams': er.get('retained_oil_grams'),
            'mass_basis': er.get('mass_basis'), 'nutrition_source': 'nutrition5k_dishes_verified',
        })
    rows = old_rows + new_rows
    counts = Counter(r['split'] for r in rows)
    valid_per_field = {name: {'count': sum(r['mask'][i] == 1 for r in rows), 'unit': UNITS[i]}
                       for i, name in enumerate(TARGETS)}
    manifest = {
        'protocol': 'nutrition5k_overhead_official_ids_macros_expanded_v1',
        'frozen_protocol': macros_manifest['protocol'],
        'target_names': TARGETS, 'target_units': UNITS, 'macro_names': MACRO_NAMES,
        'category_to_idx': macros_manifest['category_to_idx'],
        'category_source': 'derived_ingredient_mass_keyword_groups_not_official_food_classes',
        'target_stats': target_stats(rows), 'counts': dict(counts),
        'category_counts': {s: dict(Counter(r['category'] for r in rows if r['split'] == s)) for s in counts},
        'valid_per_field': valid_per_field,
        'metadata_sha256': file_digest(csv_path),
        'new_dishes_from_expanded': len(new_rows),
        'missing_handling': 'macro unknown -> value 0 mask 0; real zero -> 0 mask 1',
        'limitations': macros_manifest['limitations'],
        'rows': rows,
    }
    atomic_json(output, manifest)
    print(json.dumps({'manifest': str(output), 'sha256': file_digest(output), 'counts': dict(counts),
                      'new_dishes': len(new_rows), 'valid_per_field': valid_per_field,
                      'target_stats': manifest['target_stats']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    build()
