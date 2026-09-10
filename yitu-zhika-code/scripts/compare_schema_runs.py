"""跨 label_schema 的诚实对照（同一批 507 测试盘）。

问题：label_schema 从 11 类升到 12 类（新增 fruit）后，
- "全类别 Macro-F1"的分母不同（11 vs 12）；
- `vegetable` 的定义变了（水果类食材移出）；
所以两个模型的分类指标**不能直接并排比较**。本脚本给出三种口径，报告中必须写明用的是哪一种：

1. **各自 schema 内**：模型对自己训练时的标签体系评估（自证，不可跨模型比较）。
2. **统一到旧 schema 真值**：以 v1（11 类）标签为真值；新模型预测按类名映射回旧体系，
   预测为旧体系不存在的类别（fruit）记为错。
3. **统一到新 schema 真值**：以 v2（12 类）标签为真值；旧模型预测按类名映射到新体系，
   旧模型根本无法预测 fruit → 真值为 fruit 的餐盘必然算错。

另外如实报告"不可映射预测数"（口径 2 里的 fruit 预测）。
用法：
  python scripts/compare_schema_runs.py \
      --run v1_expanded:results/meal_macros_expanded_v1/manifest.json \
      --run v3_corrected_seed42:results/meal_macros_corrected_v2/manifest.json
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_classification_support import support_metrics   # noqa: E402
from scripts.capsicum_job import atomic_json, file_digest          # noqa: E402

PRED_DIR = ROOT / 'results/meal_macros_train_v1'


def load_schema(manifest_path):
    data = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    idx_to_name = {v: k for k, v in data['category_to_idx'].items()}
    names = [idx_to_name[i] for i in range(len(idx_to_name))]
    truth = {r['dish_id']: r['category'] for r in data['rows'] if r['split'] == 'test'}
    return {'path': str(Path(manifest_path)), 'names': names,
            'name_to_idx': {n: i for i, n in enumerate(names)}, 'truth': truth}


def load_predictions(tag):
    path = PRED_DIR / tag / 'test_predictions_category.csv'
    if not path.exists():
        raise SystemExit(f'缺少 {path}；先跑 scripts/export_test_predictions.py --tag {tag}')
    out = {}
    with path.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['dish_id']] = int(r['pred_coarse_class'])
    return out, str(path)


def evaluate(tag, manifest_path, truth_schema, pred_schema):
    """以 truth_schema 为真值评估 tag 模型（预测按类名从 pred_schema 映射到 truth_schema）。"""
    preds, path = load_predictions(tag)
    ids = [d for d in truth_schema['truth'] if d in preds]
    if len(ids) != len(truth_schema['truth']):
        raise SystemExit(f'{tag}: 预测覆盖 {len(ids)}/{len(truth_schema["truth"])} 个测试盘')
    t_idx, p_idx, unmappable = [], [], []
    for d in sorted(ids):
        truth_name = truth_schema['truth'][d]
        pred_name = pred_schema['names'][preds[d]]
        t_idx.append(truth_schema['name_to_idx'][truth_name])
        if pred_name in truth_schema['name_to_idx']:
            p_idx.append(truth_schema['name_to_idx'][pred_name])
        else:
            p_idx.append(None)
            unmappable.append(pred_name)
    n = len(t_idx)
    matches = sum(1 for t, p in zip(t_idx, p_idx) if p is not None and t == p)
    mappable = [(t, p) for t, p in zip(t_idx, p_idx) if p is not None]
    metrics = support_metrics([t for t, _ in mappable], [p for _, p in mappable],
                              truth_schema['names'], name=f'{tag}@{Path(truth_schema["path"]).name}',
                              manifest=truth_schema['path'])
    return {
        'tag': tag, 'predictions': path, 'n': n,
        'accuracy_over_all_dishes': round(matches / n, 4),
        'n_unmappable_predictions': len(mappable) and len(unmappable) or len(unmappable),
        'unmappable_classes': dict(Counter(unmappable)),
        'metrics_on_mappable_subset': {
            k: metrics[k] for k in ('n', 'accuracy', 'macro_f1_all_classes', 'macro_f1_supported',
                                    'balanced_accuracy_supported', 'n_supported_classes',
                                    'pred_distribution')},
        'per_class_on_mappable_subset': metrics['per_class'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', default=[],
                    help='tag:manifest_path —— 该模型在**自己 schema 真值**下的指标（可重复）')
    ap.add_argument('--cross', action='append', default=[],
                    help='tag:model_manifest:truth_manifest —— 把 tag 的预测按 model_manifest 解释、'
                         '在 truth_manifest 的真值下评估（可重复；用于跨 label_schema 对照）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if not args.run and not args.cross:
        raise SystemExit('至少给一个 --run 或 --cross')

    report = {'runs': {}, 'schemas': {}, 'cross_evaluations': {}}

    def schema(path, label):
        if label not in report['schemas']:
            s = load_schema(path)
            report['schemas'][label] = {'manifest': s['path'], 'manifest_sha256': file_digest(s['path']),
                                        'n_classes': len(s['names']), 'names': s['names']}
        return load_schema(path)

    for spec in args.run:
        tag, _, manifest = spec.partition(':')
        if not manifest:
            raise SystemExit(f'--run {spec} 缺少 manifest 路径')
        model_schema = schema(manifest, f'{tag}_own')
        report['runs'][tag] = evaluate(tag, manifest, model_schema, model_schema)

    for spec in args.cross:
        parts = spec.split(':')
        if len(parts) != 3:
            raise SystemExit(f'--cross {spec} 需要 tag:model_manifest:truth_manifest')
        tag, model_manifest, truth_manifest = parts
        model_schema = schema(model_manifest, f'{tag}_model')
        truth_schema = schema(truth_manifest, 'truth_' + Path(truth_manifest).parent.name)
        key = f'{tag}_evaluated_on_{Path(truth_manifest).parent.name}'
        report['cross_evaluations'][key] = evaluate(tag, model_manifest, truth_schema, model_schema)

    out = Path(args.out) if args.out else ROOT / 'artifacts/schema-comparison.json'
    atomic_json(out, report)
    print(f'证据: {out}')
    print(json.dumps({'own_schema': {t: {k: report['runs'][t][k] for k in
                                         ('n', 'accuracy_over_all_dishes')} for t in report['runs']},
                      'own_metrics': {t: report['runs'][t]['metrics_on_mappable_subset']
                                      for t in report['runs']},
                      'cross': {k: {kk: v[kk] for kk in ('n', 'accuracy_over_all_dishes',
                                                         'n_unmappable_predictions',
                                                         'unmappable_classes')}
                                for k, v in report['cross_evaluations'].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
