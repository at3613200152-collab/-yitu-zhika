"""对比基线 / §7-B 非负输出 / §7-C 类别加权 CE 三个运行（冻结 507），并给逐类召回对比。

用法：python scripts/compare_p7_runs.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_classification_support import support_metrics, categories_from_manifest  # noqa: E402

RUNS = [('v1_expanded', '基线(11类,3390)'), ('v5_nonneg_seed42', 'B 非负输出'),
        ('v5_clsw_seed42', 'C 类别加权CE')]
FIELDS = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']
MANIFEST = ROOT / 'results/meal_macros_expanded_v1/manifest.json'

print('== 回归与异常值（冻结 507）==')
print(f"{'run':18s}{'说明':16s}{'kcal':>7s}{'mass':>7s}{'prot':>7s}{'carb':>7s}{'fat':>7s}{'acc':>8s}{'neg(k/m/p/c/f)':>18s}{'best_ep':>9s}")
rows = {}
for tag, note in RUNS:
    p = ROOT / f'results/meal_macros_train_v1/{tag}/status.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    t, pf = d['test'], d['test']['per_field']
    neg = '/'.join(str(pf[k]['neg']) for k in FIELDS)
    rows[tag] = d
    print(f"{tag:18s}{note:16s}{pf['calories']['mae']:>7.2f}{pf['mass']['mae']:>7.2f}"
          f"{pf['protein']['mae']:>7.2f}{pf['carbohydrate']['mae']:>7.2f}{pf['fat']['mae']:>7.2f}"
          f"{t['coarse_category_accuracy']:>8.4f}{neg:>18s}{d.get('best_epoch'):>9d}")

print()
print('== 分类：Macro-F1 / 平衡召回 / 逐类召回 ==')
cats = categories_from_manifest(str(MANIFEST))
per_run = {}
for tag, note in RUNS:
    # v1_expanded 的 CSV 早于"补类别列"的改动，需用导出脚本生成的带类别版本
    cand = ROOT / f'results/meal_macros_train_v1/{tag}/test_predictions_category.csv'
    main = ROOT / f'results/meal_macros_train_v1/{tag}/test_predictions.csv'
    with main.open(encoding='utf-8') as f:
        has_cols = 'true_coarse_class' in (f.readline() or '')
    csv_path = main if has_cols else cand
    true, pred = [], []
    with csv_path.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            true.append(int(r['true_coarse_class']))
            pred.append(int(r['pred_coarse_class']))
    m = support_metrics(true, pred, cats, name=tag, manifest=str(MANIFEST))
    per_run[tag] = m
    print(f"{tag:18s} acc={m['accuracy']:.4f}  macroF1(有支持)={m['macro_f1_supported']:.4f}  "
          f"macroF1(全标签)={m['macro_f1_all_classes']:.4f}  平衡召回={m['balanced_accuracy_supported']:.4f}")

print()
hdr = f"{'类别':18s}{'support':>8s}" + ''.join(f'{t.replace("_seed42",""):>16s}' for t, _ in RUNS)
print(hdr)
for i, c in enumerate(cats):
    line = f'{c:18s}{per_run["v1_expanded"]["per_class"][i]["support"]:>8d}'
    for tag, _ in RUNS:
        r = per_run[tag]['per_class'][i]['recall']
        line += f'{("None" if r is None else format(r, ".3f")):>16s}'
    print(line)

out = ROOT / 'artifacts/p7-bc-comparison.json'
out.write_text(json.dumps({t: {'regression': rows[t]['test']['per_field'],
                               'acc': rows[t]['test']['coarse_category_accuracy'],
                               'classification': per_run[t]} for t, _ in RUNS},
                          ensure_ascii=False, indent=1), encoding='utf-8')
print(f'\n证据: {out}')
