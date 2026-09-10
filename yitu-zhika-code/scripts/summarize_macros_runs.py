"""多种子/多运行汇总：把各 tag 的 507 测试指标列成表，并按分组给出均值与极差。

用于回答"某改动是否超出运行间噪声"：只有分组区间不重叠（或差异远大于组内极差）才可宣称差异。
只读 results/meal_macros_train_v1/*/status.json。

用法：
  python scripts/summarize_macros_runs.py \
      --group old_label=v1_expanded,v2_seed42 --group corrected=v3_corrected_seed42,v3_corrected_seed43
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest   # noqa: E402

RUNS = ROOT / 'results/meal_macros_train_v1'
FIELDS = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']


def load_run(tag):
    folder = RUNS / tag
    status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    if status.get('stage') != 'complete' or 'test' not in status:
        return {'tag': tag, 'incomplete': True, 'stage': status.get('stage')}
    test = status['test']
    out = {'tag': tag, 'stage': 'complete', 'n': test.get('n'),
           'coarse_category_accuracy': test.get('coarse_category_accuracy'),
           'best_epoch': status.get('best_epoch'),
           'checkpoint_sha256': file_digest(ROOT / 'checkpoints/meal_macros_v1' / tag / 'best.pt')
           if (ROOT / 'checkpoints/meal_macros_v1' / tag / 'best.pt').exists() else None}
    for f in FIELDS:
        pf = test['per_field'][f]
        out[f] = {'mae': pf['mae'], 'rmse': pf['rmse'], 'neg': pf['neg'], 'n': pf['n']}
    return out


def stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {'n': len(vals), 'mean': round(sum(vals) / len(vals), 3),
            'min': round(min(vals), 3), 'max': round(max(vals), 3),
            'range': round(max(vals) - min(vals), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', action='append', default=[],
                    help='name=tag1,tag2（可重复）')
    ap.add_argument('--tag', action='append', default=[], help='单独列出某个 tag')
    ap.add_argument('--out', default='artifacts/macros-runs-summary.json')
    args = ap.parse_args()

    groups = {}
    for spec in args.group:
        name, _, tags = spec.partition('=')
        groups[name] = [t for t in tags.split(',') if t]

    report = {'runs': {}, 'groups': {}}
    for tag in args.tag + [t for tags in groups.values() for t in tags]:
        if tag not in report['runs']:
            report['runs'][tag] = load_run(tag)
    for name, tags in groups.items():
        report['groups'][name] = {
            'tags': tags,
            'metrics': {f: stats([report['runs'][t].get(f, {}).get('mae') for t in tags
                                  if not report['runs'][t].get('incomplete')]) for f in FIELDS},
            'coarse_category_accuracy': stats([report['runs'][t].get('coarse_category_accuracy')
                                               for t in tags]),
            'negatives': {f: stats([report['runs'][t].get(f, {}).get('neg') for t in tags
                                    if not report['runs'][t].get('incomplete')]) for f in FIELDS},
        }

    atomic_json(Path(args.out), report)
    print(f'证据: {args.out}')
    header = '{:<24s}{:>8s}'.format('tag', 'best_ep') + ''.join(f'{f[:6]:>10s}' for f in FIELDS) \
        + f'{"acc":>8s}' + ''.join(f'{f[:4]+"neg":>9s}' for f in FIELDS)
    print(header)
    for tag, r in report['runs'].items():
        if r.get('incomplete'):
            print('{:<24s}{:>8s}  (未完成: {})'.format(tag, '-', r.get('stage')))
            continue
        line = '{:<24s}{:>8s}'.format(tag, str(r['best_epoch']))
        line += ''.join(f'{r[f]["mae"]:>10.2f}' for f in FIELDS)
        line += f'{r["coarse_category_accuracy"]:>8.4f}'
        line += ''.join(f'{r[f]["neg"]:>9d}' for f in FIELDS)
        print(line)
    for name, g in report['groups'].items():
        line = f'[{name}] n={g["metrics"]["calories"]["n"]} '
        line += ' '.join(f'{f}={g["metrics"][f]["mean"]}(±{g["metrics"][f]["range"]/2:.2f})'
                         for f in FIELDS)
        if g['coarse_category_accuracy']:
            line += f' acc={g["coarse_category_accuracy"]["mean"]}'
        print(line)


if __name__ == '__main__':
    main()
