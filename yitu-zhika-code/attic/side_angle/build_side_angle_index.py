"""生成 Nutrition5k 侧视图索引表，便于人工检验与比对。

输出 results/side_angle_index.csv（128 行）：
  dish_id, category, split, gt_calories, gt_mass, gt_protein, gt_carb, gt_fat,
  train_image(训练时实际读入的图，即 pilot 的 frame_0000),
  pilot_frames(pilot 目录下 3 帧), expansion_frames(expansion 目录下 3 帧),
  frame_used_in_training(是否为该图的训练帧)

要点（已实测）：这 128 道菜的 `image` 就是侧视帧且 split=train，
所以 frame_0000 属训练图；frame_0030/0060 是同菜同场次的其他帧（视角鲁棒性检验，不是新数据泛化）。
用法：python scripts/build_side_angle_index.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MACROS_MANIFEST = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
SIDE_MANIFEST = ROOT / 'data/side_angle_manifest/manifest.json'
OUT = ROOT / 'results/side_angle_index.csv'
PILOT = 'D:/yitu-data/Nutrition5k/side_angle_pilot_v1'
EXPANSION = 'D:/yitu-data/Nutrition5k/side_angle_expansion_v1'


def main():
    rows = json.loads(MACROS_MANIFEST.read_text(encoding='utf-8'))['rows']
    with_views = {r['dish_id']: r for r in rows if r.get('views')}
    side = {r['dish_id']: r for r in json.loads(SIDE_MANIFEST.read_text(encoding='utf-8'))['rows']}

    out_rows = []
    for dish_id, r in sorted(with_views.items()):
        t = r['targets']
        pilot_frames = [v['path'].replace('\\', '/') for v in r['views']]
        exp = side.get(dish_id) or {}
        exp_frames = [f['path'] for f in exp.get('frames', [])]
        train_image = (r.get('image') or '').replace('\\', '/')
        out_rows.append({
            'dish_id': dish_id, 'category': r['category'], 'split': r['split'],
            'gt_calories': t[0], 'gt_mass': t[1], 'gt_protein': t[2],
            'gt_carb': t[3], 'gt_fat': t[4],
            'train_image': train_image,
            'pilot_frames': ';'.join(pilot_frames),
            'expansion_frames': ';'.join(exp_frames),
            'frame_used_in_training': train_image,
            'expansion_dir_exists': (Path(exp_frames[0]).parent.exists() if exp_frames else False),
            'pilot_dir_exists': (Path(pilot_frames[0]).parent.exists() if pilot_frames else False),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f'索引已写出: {OUT}（{len(out_rows)} 行）')
    print(f'pilot 目录可用: {sum(1 for r in out_rows if r["pilot_dir_exists"])}/{len(out_rows)}；'
          f'expansion 目录可用: {sum(1 for r in out_rows if r["expansion_dir_exists"])}/{len(out_rows)}')
    print('前 3 行:')
    for r in out_rows[:3]:
        print(' ', r['dish_id'], r['category'], 'GT', r['gt_calories'], 'kcal /', r['gt_mass'], 'g')
        print('   训练图:', r['train_image'])
        print('   expansion帧:', r['expansion_frames'])
    print('\n分类分布:', dict(sorted({c: sum(1 for r in out_rows if r['category'] == c)
                                      for c in {r['category'] for r in out_rows}}.items())))


if __name__ == '__main__':
    main()
