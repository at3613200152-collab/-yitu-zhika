"""P1-B：五目标（卡路里/重量/蛋白/碳水/脂肪）manifest 构建。

- 复用官方冻结划分（results/meal_official_v1/manifest.json）的 dish_id/split/image/溯源 + 已有审计，
  不重新划分，避免改动冻结标签/划分。
- 从 data/Nutrition5k/dishes_verified.csv 按字段名读 total_calories/total_mass/total_fat/total_carb/total_protein。
- 逐字段有效 mask：字段为有限且 >=0 才有效；缺失/无效记为 mask=0，**不补 0**（真实 0 保留为 0 且 mask=1）。
- target_stats 用 train-only 逐字段统计（只统计该字段 mask=1 的样本）。
- 输出 results/meal_macros_v1/manifest.json（新 protocol，不覆盖冻结官方清单）。
用法：python src/data/meal_macros_manifest.py
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

PROTOCOL = 'nutrition5k_overhead_official_ids_macros_v1'
TARGETS = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']
UNITS = ['kcal', 'g', 'g', 'g', 'g']
# 字段名映射（dishes_verified.csv -> 目标列）
CSV_FIELDS = ['total_calories', 'total_mass', 'total_protein', 'total_carb', 'total_fat']
# 宏量字段的外部/接口命名
MACRO_NAMES = {'protein': 'protein_g', 'carbohydrate': 'carbohydrate_g', 'fat': 'fat_g'}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


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
        out[name] = {'mean': round(mean, 4), 'std': round(std, 4), 'samples': len(vals),
                     'unit': UNITS[i], 'source': 'train_only'}
    return out


def build():
    frozen = ROOT / 'results/meal_official_v1/manifest.json'
    csv_path = ROOT / 'data/Nutrition5k/dishes_verified.csv'
    audit_path = ROOT / 'results/metadata_audit/nutrition5k_audit.json'
    output = ROOT / 'results/meal_macros_v1/manifest.json'

    if file_digest(csv_path) != json.loads(audit_path.read_text(encoding='utf-8'))['verified_csv_sha256']:
        raise ValueError('Verified metadata differs from audited hash')

    frozen_rows = json.loads(frozen.read_text(encoding='utf-8'))['rows']
    with csv_path.open(encoding='utf-8', newline='') as f:
        macro_by_dish = {r['dish_id']: r for r in csv.DictReader(f)}
    if len(macro_by_dish) == 0:
        raise ValueError('Empty verified CSV')

    rows = []
    seen_pixels = {}
    for row in frozen_rows:
        macro = macro_by_dish.get(row['dish_id'])
        if macro is None:
            continue
        raw = [_f(macro.get(c)) for c in CSV_FIELDS]  # calories, mass, protein, carb, fat
        if raw[0] is None or raw[1] is None:
            raise ValueError(f'Missing calories/mass for {row["dish_id"]}')
        targets = [raw[0], raw[1],
                   raw[2] if raw[2] is not None and raw[2] >= 0 else 0.0,
                   raw[3] if raw[3] is not None and raw[3] >= 0 else 0.0,
                   raw[4] if raw[4] is not None and raw[4] >= 0 else 0.0]
        # mask：字段有效为 1；宏量未知时值用 0 占位但 mask=0（不当作真实 0 参与损失）
        mask = [1, 1] + [1 if raw[i] is not None and raw[i] >= 0 else 0 for i in (2, 3, 4)]
        if raw[1] <= 0:
            raise ValueError(f'Invalid mass: {row["dish_id"]}')
        # 沿用 frozen 的像素查重信息（image_sha256 溯源已含）
        new_row = {
            'dish_id': row['dish_id'], 'split': row['split'], 'capture_day_utc': row['capture_day_utc'],
            'image': row['image'], 'image_sha256': row['image_sha256'], 'pixel_sha256': row['pixel_sha256'],
            'size': row['size'], 'targets': targets, 'mask': mask, 'category': row['category'],
            'category_idx': row['category_idx'], 'nutrition_source': 'nutrition5k_dishes_verified',
        }
        rows.append(new_row)

    counts = Counter(r['split'] for r in rows)
    category_to_idx = json.loads(frozen.read_text(encoding='utf-8'))['category_to_idx']
    valid_per_field = {}
    for i, name in enumerate(TARGETS):
        valid_per_field[name] = {'count': sum(r['mask'][i] == 1 for r in rows), 'unit': UNITS[i]}

    manifest = {
        'protocol': PROTOCOL,
        'frozen_protocol': json.loads(frozen.read_text(encoding='utf-8'))['protocol'],
        'target_names': TARGETS,
        'target_units': UNITS,
        'macro_names': MACRO_NAMES,
        'category_to_idx': category_to_idx,
        'category_source': 'derived_ingredient_mass_keyword_groups_not_official_food_classes',
        'target_stats': target_stats(rows),
        'counts': dict(counts),
        'category_counts': {s: dict(Counter(r['category'] for r in rows if r['split'] == s)) for s in counts},
        'valid_per_field': valid_per_field,
        'missing_handling': 'macro unknown -> value 0 placeholder, mask 0 (excluded from loss); real zero -> 0 with mask 1',
        'metadata_sha256': file_digest(csv_path),
        'limitations': json.loads(frozen.read_text(encoding='utf-8'))['limitations'],
        'rows': rows,
    }
    atomic_json(output, manifest)
    print(json.dumps({'manifest': str(output), 'sha256': file_digest(output), 'counts': dict(counts),
                      'target_stats': manifest['target_stats'], 'valid_per_field': valid_per_field}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.parse_args()
    build()
