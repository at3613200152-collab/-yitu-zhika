"""端到端追溯：从某个被修正餐盘出发，追到"修正规则 → 新标签 → 新 manifest → 训练配置 → 权重"。

对应整改清单 §4 验收口径：「从某个被修正餐盘出发，能追踪修正规则→新标签→新manifest→训练配置→权重」。

用法：
  python scripts/trace_label_change.py                       # 自动挑一个标签变更且与水果有关的餐盘
  python scripts/trace_label_change.py --dish dish_1561662216 --ckpt v3_corrected_seed42
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import file_digest                       # noqa: E402
from src.data.build_categories import match_ingredient             # noqa: E402

OLD_MANIFEST = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
NEW_MANIFEST = ROOT / 'results/meal_macros_corrected_v2/manifest.json'
CHANGES = ROOT / 'results/meal_macros_corrected_v2/label_changes.json'
INGREDIENTS = ROOT / 'data/Nutrition5k/dish_ingredients.csv'
RULE_CODE = ROOT / 'src/data/build_categories.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pick_dish(changes, new_rows):
    """优先挑"新标签为 fruit"的变更餐盘（最能体现 schema 升级）。"""
    for dish_id, info in sorted(changes['changes'].items()):
        if info.get('new') == 'fruit':
            return dish_id
    return sorted(changes['changes'])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dish', default=None)
    ap.add_argument('--ckpt', default='v3_corrected_seed42',
                    help='checkpoints/meal_macros_v1/<ckpt>/best.pt（存在才输出训练配置段）')
    args = ap.parse_args()

    changes = read(CHANGES)
    old_rows = {r['dish_id']: r for r in read(OLD_MANIFEST)['rows']}
    new_rows = {r['dish_id']: r for r in read(NEW_MANIFEST)['rows']}
    dish = args.dish or pick_dish(changes, new_rows)
    if dish not in new_rows:
        raise SystemExit(f'{dish} 不在修正清单里')

    ingredients = []
    with INGREDIENTS.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['dish_id'] != dish:
                continue
            cat, kw = match_ingredient(r['ingr_name'])
            ingredients.append({'ingredient': r['ingr_name'], 'grams': float(r['grams'] or 0),
                                'rule_category': cat, 'matched_keyword': kw})
    total = sum(i['grams'] for i in ingredients)
    weight_by_cat = {}
    for i in ingredients:
        weight_by_cat[i['rule_category']] = weight_by_cat.get(i['rule_category'], 0.0) + i['grams']

    report = {
        'dish_id': dish,
        '1_ingredients_and_rule_hits': sorted(ingredients, key=lambda x: -x['grams']),
        '2_mass_aggregation': {
            'total_grams': round(total, 3),
            'per_category_grams': {k: round(v, 3) for k, v in sorted(weight_by_cat.items(), key=lambda kv: -kv[1])},
            'threshold_rule': '最高类别占比 >= 40% 取该类，否则 mixed',
        },
        '3_label_old_new': {
            'old': old_rows[dish]['category'], 'new': new_rows[dish]['category'],
            'new_category_idx': new_rows[dish]['category_idx'],
            'change_record': changes['changes'].get(dish),
            'split': new_rows[dish]['split'],
        },
        '4_provenance': {
            'label_rule_code': str(RULE_CODE.relative_to(ROOT)),
            'label_rule_code_sha256': file_digest(RULE_CODE),
            'old_manifest_sha256': file_digest(OLD_MANIFEST),
            'new_manifest': str(NEW_MANIFEST.relative_to(ROOT)),
            'new_manifest_sha256': file_digest(NEW_MANIFEST),
            'label_schema_version': read(NEW_MANIFEST).get('label_schema_version'),
            'ingredients_sha256': read(NEW_MANIFEST).get('ingredients_sha256'),
        },
    }

    ckpt = ROOT / 'checkpoints/meal_macros_v1' / args.ckpt / 'best.pt'
    if ckpt.exists():
        import torch
        cfg = torch.load(ckpt, map_location='cpu', weights_only=True)['config']
        report['5_training_config'] = {
            'checkpoint': str(ckpt.relative_to(ROOT)),
            'checkpoint_sha256': file_digest(ckpt),
            'tag': cfg.get('tag'), 'seed': cfg.get('seed'), 'num_classes': cfg.get('num_classes'),
            'label_schema_version': cfg.get('label_schema_version'),
            'manifest': cfg.get('manifest'), 'manifest_sha256': cfg.get('manifest_sha256'),
            'manifest_sha_matches_current_file': cfg.get('manifest_sha256') == report['4_provenance']['new_manifest_sha256'],
            'code_sha256': cfg.get('code_sha256'),
        }
    else:
        report['5_training_config'] = f'未找到 {ckpt}（训练尚未产出或用其他 tag）'

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
