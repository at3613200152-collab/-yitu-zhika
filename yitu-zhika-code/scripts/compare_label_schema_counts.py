"""对比旧(11类)与新(12类)修正清单的类别分布（train/val/test）。
只读，不写产物。用法：python scripts/compare_label_schema_counts.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
NEW = ROOT / 'results/meal_macros_corrected_v2/manifest.json'


def counts(path):
    rows = json.loads(path.read_text(encoding='utf-8'))['rows']
    out = {}
    for split in ('train', 'val', 'test'):
        c = Counter(r['category'] for r in rows if r['split'] == split)
        tot = sum(c.values())
        out[split] = (c, tot)
    return out


def main():
    old, new = counts(OLD), counts(NEW)
    cats = sorted(set(old['train'][0]) | set(new['train'][0]))
    print('train 分布对比:')
    print('  {:<16s} {:>6s} {:>7s} {:>6s} {:>7s}'.format('category', 'old', 'old%', 'new', 'new%'))
    for c in sorted(cats, key=lambda x: -new['train'][0].get(x, 0)):
        o, n = old['train'][0].get(c, 0), new['train'][0].get(c, 0)
        print('  {:<16s} {:6d} {:6.1f}% {:6d} {:6.1f}%'.format(
            c, o, 100 * o / old['train'][1], n, 100 * n / new['train'][1]))
    print('  total', old['train'][1], new['train'][1])
    for split in ('val', 'test'):
        print(f'{split}: old_total={old[split][1]} new_total={new[split][1]}  新清单有支持的类别数='
              f'{len([k for k, v in new[split][0].items() if v > 0])}')


if __name__ == '__main__':
    main()
