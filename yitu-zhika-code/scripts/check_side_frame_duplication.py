"""检查侧视三帧之间的相似度，判断"未参与训练的帧"是否真的算新样本。

结论用途：若 0000 与 0030/0060 近乎重复，则用 0030/0060 做"泛化/鲁棒性"检验不成立，
只能说明"近乎同一张图上的记忆效果"。参考量级：完全不同的两张图平均像素绝对差通常 > 0.15。

用法：python scripts/check_side_frame_duplication.py --limit 0   # 0=全部 128 道
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json                     # noqa: E402

INDEX = ROOT / 'results/side_angle_index.csv'
SIZE = (256, 256)


def frames_of(row):
    out = {}
    for p in row['expansion_frames'].split(';') + row['pilot_frames'].split(';'):
        if p:
            out[Path(p).stem.split('_')[-1]] = p
    return out


def arr(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='0=全部')
    ap.add_argument('--out', default=str(ROOT / 'artifacts/side-frame-duplication.json'))
    args = ap.parse_args()

    rows = list(csv.DictReader(INDEX.open(encoding='utf-8')))
    if args.limit:
        rows = rows[:args.limit]

    per_dish, skipped = {}, []
    for r in rows:
        f = frames_of(r)
        if not all(k in f for k in ('0000', '0030', '0060')):
            skipped.append(r['dish_id'])
            continue
        a, b, c = arr(f['0000']), arr(f['0030']), arr(f['0060'])
        per_dish[r['dish_id']] = {
            'diff_0000_vs_0030': float(np.abs(a - b).mean()),
            'diff_0000_vs_0060': float(np.abs(a - c).mean()),
            'diff_0030_vs_0060': float(np.abs(b - c).mean()),
        }

    def summarize(key):
        vals = sorted(v[key] for v in per_dish.values())
        n = len(vals)
        return {'n': n, 'mean': sum(vals) / n, 'median': vals[n // 2],
                'p90': vals[int(0.9 * (n - 1))], 'max': vals[-1]}

    report = {'n_dishes': len(per_dish), 'skipped_no_3_frames': skipped,
              'summary': {k: summarize(k) for k in ('diff_0000_vs_0030', 'diff_0000_vs_0060',
                                                    'diff_0030_vs_0060')},
              'reference': '完全不同的两张图平均像素绝对差通常 > 0.15（[0,1] 基）',
              'per_dish': per_dish}
    atomic_json(Path(args.out), report)
    print(json.dumps({'n_dishes': report['n_dishes'],
                      'skipped': len(report['skipped_no_3_frames']),
                      'summary': report['summary'], 'reference': report['reference']},
                     ensure_ascii=False, indent=2))
    print(f'证据: {args.out}')


if __name__ == '__main__':
    main()
