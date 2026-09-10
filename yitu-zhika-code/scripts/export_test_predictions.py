"""导出五头模型的 507 测试集逐盘预测（含类别），用于逐类支持度审计。

背景：`train_meal_macros.py` 写出的 `test_predictions.csv` 只有五目标数值列，
没有 true/pred 类别列，因此 `scripts/audit_classification_support.py` 无法用于五头模型，
也无法做"同一真值、不同 label_schema"的对照。

本脚本**不改动训练模块**（避免在训练进行中改动 code_sha256 污染在跑运行的配置哈希），
自带一个测试集推理循环，输出：
  results/meal_macros_train_v1/<tag>/test_predictions_category.csv
  results/meal_macros_train_v1/<tag>/test_category_metrics.json

用法：
  python scripts/export_test_predictions.py \
      --ckpt checkpoints/meal_macros_v1/v1_expanded/best.pt \
      --manifest results/meal_macros_expanded_v1/manifest.json \
      --tag v1_expanded --device cpu
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest          # noqa: E402
from src.models.meal_macros_net import MealMacrosNet, TARGETS      # noqa: E402
from src.training.train_meal_macros import MacrosDataset, per_field_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    idx_to_name = {v: k for k, v in manifest['category_to_idx'].items()}
    device = torch.device(args.device)

    model = MealMacrosNet(manifest, pretrained=False)
    payload = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()

    data = MacrosDataset(manifest, 'test', 0)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=False,
                                         num_workers=0, pin_memory=False)
    rows, truths, preds, masks, tcls, pcls = [], [], [], [], [], []
    with torch.inference_mode():
        for images, target, mask, cls, dish_ids in loader:
            images = images.to(device)
            if device.type == 'cuda':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, prediction = model(images)
            else:
                logits, prediction = model(images)
            prediction = prediction.detach().float().cpu()
            if not torch.isfinite(prediction).all() or not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite model output during export')
            top = logits.float().argmax(1).detach().cpu()
            truths.append(target)
            preds.append(prediction)
            masks.append(mask)
            tcls.extend(cls.tolist())
            pcls.extend(top.tolist())
            for i, dish_id in enumerate(dish_ids):
                rows.append({'dish_id': dish_id,
                             'true_coarse_class': int(cls[i]), 'pred_coarse_class': int(top[i])})

    truths_t = torch.cat(truths)
    preds_t = torch.cat(preds)
    masks_t = torch.cat(masks)
    per_field = per_field_metrics(truths_t.numpy(), preds_t.numpy(), masks_t.numpy())
    accuracy = sum(a == b for a, b in zip(tcls, pcls)) / len(tcls)

    out_dir = ROOT / 'results/meal_macros_train_v1' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'test_predictions_category.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dish_id', 'true_coarse_class', 'pred_coarse_class',
                    'true_coarse_name', 'pred_coarse_name']
                   + [f'true_{t}' for t in TARGETS] + [f'pred_{t}' for t in TARGETS]
                   + [f'mask_{t}' for t in TARGETS])
        for i, r in enumerate(rows):
            w.writerow([r['dish_id'], r['true_coarse_class'], r['pred_coarse_class'],
                        idx_to_name.get(r['true_coarse_class']), idx_to_name.get(r['pred_coarse_class'])]
                       + [f'{float(truths_t[i, j]):.6f}' for j in range(len(TARGETS))]
                       + [f'{float(preds_t[i, j]):.6f}' for j in range(len(TARGETS))]
                       + [int(masks_t[i, j]) for j in range(len(TARGETS))])

    summary = {
        'tag': args.tag, 'checkpoint': str(Path(args.ckpt)),
        'checkpoint_sha256': file_digest(args.ckpt),
        'manifest': str(manifest_path), 'manifest_sha256': file_digest(manifest_path),
        'label_schema_version': manifest.get('label_schema_version', 'v1_11class'),
        'n_classes': len(manifest['category_to_idx']),
        'device': str(device), 'n': len(rows),
        'coarse_category_accuracy': round(accuracy, 6),
        'per_field': per_field,
        'predictions_csv': str(csv_path.relative_to(ROOT)),
        'note': '本文件由 scripts/export_test_predictions.py 生成；类别索引按 --manifest 解释。',
    }
    atomic_json(out_dir / 'test_category_metrics.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
