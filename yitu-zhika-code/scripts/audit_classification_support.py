"""§十 验收门槛：分类支持度审计。

从已有的 test_predictions.csv（true_coarse_class / pred_coarse_class）计算：
  逐类 support/predicted/TP/FP/FN、precision/recall/F1、
  全标签集合与"有支持类别"两套 Macro-F1、balanced accuracy、accuracy、11x11 混淆矩阵。
写入 artifacts/classification-support-<stamp>/evidence.json 并打印摘要。

用法：python scripts/audit_classification_support.py \
        --predictions results/meal_rgb_official_v1/test_predictions.csv --name rgb
可选：--manifest results/meal_macros_corrected_v2/manifest.json  （label_schema v2 为 12 类，含 fruit；
      不传则用默认 11 类顺序，仅适用于旧 11 类产物）
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_CATEGORIES = ['dairy', 'dessert', 'egg', 'grain', 'meat', 'mixed', 'other',
                      'sauce_condiment', 'seafood', 'soup_stew', 'vegetable']


def categories_from_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    cti = data['category_to_idx']
    return [c for c in sorted(cti, key=cti.get)]


def support_metrics(true, pred, categories, *, name='model', manifest=None):
    """逐类 support/predicted/TP/FP/FN + P/R/F1、两套 Macro-F1、balanced accuracy、混淆矩阵。

    供本脚本与 scripts/compare_schema_runs.py 共用（同一套定义，避免两处实现漂移）。
    """
    n_cls = len(categories)
    if true and max(max(true), max(pred)) >= n_cls:
        raise ValueError(f'类别索引超出 {n_cls} 类（{categories}）；请用对应 manifest 解释')
    cm = [[0] * n_cls for _ in range(n_cls)]
    for t, p in zip(true, pred):
        cm[t][p] += 1
    n = len(true)

    per_class = []
    for i, cat in enumerate(categories):
        support = sum(cm[i])                              # 真值为 i（行和）
        predicted = sum(row[i] for row in cm)             # 预测为 i（列和）
        tp = cm[i][i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None        # 无支持即 None(不宣称)
        f1 = (2 * precision * recall / (precision + recall)) if (precision > 0 and recall) else 0.0
        per_class.append({'name': cat, 'support': support, 'predicted': predicted,
                          'tp': tp, 'precision': round(precision, 4),
                          'recall': round(recall, 4) if recall is not None else None,
                          'f1': round(f1, 4) if recall is not None else 0.0})

    supported = [c for c in per_class if c['support'] > 0]
    macro_f1_all = sum(c['f1'] for c in per_class) / n_cls
    macro_f1_supported = sum(c['f1'] for c in supported) / len(supported) if supported else None
    balanced_acc = sum(c['recall'] for c in supported) / len(supported) if supported else None
    accuracy = sum(cm[i][i] for i in range(n_cls)) / n if n else None
    pred_dist = {categories[i]: sum(cm[r][i] for r in range(n_cls)) for i in range(n_cls)}
    return {
        'name': name, 'n': n, 'n_classes': n_cls,
        'categories': list(categories), 'manifest': manifest,
        'accuracy': round(accuracy, 4) if accuracy is not None else None,
        'macro_f1_all_classes': round(macro_f1_all, 4),
        'macro_f1_supported': round(macro_f1_supported, 4) if macro_f1_supported is not None else None,
        'balanced_accuracy_supported': round(balanced_acc, 4) if balanced_acc is not None else None,
        'n_supported_classes': len(supported),
        'pred_distribution': pred_dist,
        'per_class': per_class, 'confusion_matrix': cm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--name', default='model')
    ap.add_argument('--manifest', default=None,
                    help='按该 manifest 的 category_to_idx 顺序解释类别索引（12 类必须传）')
    args = ap.parse_args()

    CATEGORIES = categories_from_manifest(args.manifest) if args.manifest else DEFAULT_CATEGORIES
    n_cls = len(CATEGORIES)
    true = []
    pred = []
    with open(args.predictions, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            true.append(int(r['true_coarse_class']))
            pred.append(int(r['pred_coarse_class']))
    summary = support_metrics(true, pred, CATEGORIES, name=args.name, manifest=args.manifest)
    per_class = summary['per_class']
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = Path(f'artifacts/classification-support-{stamp}/evidence.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'证据写入: {out}')
    print(json.dumps({k: summary[k] for k in ('name','n','accuracy','macro_f1_all_classes',
                                              'macro_f1_supported','balanced_accuracy_supported',
                                              'n_supported_classes')}, ensure_ascii=False, indent=2))
    print('per_class:')
    for c in per_class:
        print(f"  {c['name']:16s} sup={c['support']:3d} pred={c['predicted']:3d} P={c['precision']:.3f} R={c['recall']} F1={c['f1']:.3f}")


if __name__ == '__main__':
    main()
