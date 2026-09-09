"""P1-B：五目标（卡路里/重量/蛋白/碳水/脂肪）真实训练。

- 数据：results/meal_macros_v1/manifest.json（五目标 + mask + train-only 统计）
- 模型：MealMacrosNet（ResNet50, num_regression_targets=5, ImageNet 预训练骨干）
- 损失：宏量带 mask 的逐目标归一化 L1 + 0.2*CE（缺失维度 mask=0 不参与）
- 选择：验证集 reg_normalized_l1（仅在有效目标上）；冻结后 507 测试评估
- 输出：checkpoints/meal_macros_v1/<tag>/best|last.pt ; results/meal_macros_train_v1/<tag>/...
- 从旧骨干迁移：features 复用（ImageNet）；回归头&分类头默认新初始化（MealMacrosNet 新建头），
  可用 --init-from 加载官方 2 目标权重并仅复用特征层。
用法（GPU）：
  python src/training/train_meal_macros.py --tag v1 --seed 42
  python src/training/train_meal_macros.py --smoke --tag smoke   # 快速验证
"""
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest
from src.training.train_meal_official import MealDataset, set_base_seed, seed_epoch, MEAN, STD
from src.models.meal_macros_net import MealMacrosNet, macros_losses, TARGETS

MANIFEST = ROOT / 'results/meal_macros_v1/manifest.json'
WEIGHTS_DIR = ROOT / 'checkpoints/meal_macros_v1'
RESULT_DIR = ROOT / 'results/meal_macros_train_v1'


class MacrosDataset(MealDataset):
    def __getitem__(self, index):
        rgb, targets, category, dish_id = super().__getitem__(index)
        mask = torch.tensor(self.rows[index]['mask'], dtype=torch.float32)
        return rgb, targets, mask, category, dish_id


def per_field_metrics(targets, preds, masks):
    out = {}
    for i, name in enumerate(TARGETS):
        valid = masks[:, i] == 1
        y, p = targets[valid, i], preds[valid, i]
        if len(y) == 0:
            out[name] = {'n': 0, 'mae': None, 'rmse': None}
            continue
        err = p - y
        out[name] = {'n': int(len(y)), 'mae': float(np.abs(err).mean()),
                     'rmse': float(np.sqrt((err ** 2).mean())),
                     'neg': int((p < 0).sum())}
    return out


def epoch_pass(model, manifest, split, epoch, args, optimizer=None):
    training = optimizer is not None
    if training != (split == 'train'):
        raise ValueError('Optimizer may only see training rows')
    model.train(training)
    loader = DataLoader(MacrosDataset(manifest, split, epoch), batch_size=args.batch,
                        shuffle=training, num_workers=0, pin_memory=True, drop_last=False,
                        generator=torch.Generator().manual_seed(42 + epoch))
    total, reg_total, n = 0.0, 0.0, 0
    truths, preds, masks, classes, pclasses, ids = [], [], [], [], [], []
    for batch, (images, target, mask, cls, dish_ids) in enumerate(loader, 1):
        images, target, mask, cls = images.cuda(), target.cuda(), mask.cuda(), cls.cuda()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, prediction = model(images)
            loss, reg = macros_losses(logits, prediction, target, cls, model.target_scale, mask)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        size = len(dish_ids)
        n += size
        total += loss.item() * size
        reg_total += reg.item() * size
        truths.append(target.detach().cpu())
        preds.append(prediction.detach().float().cpu())
        masks.append(mask.detach().cpu())
        classes.extend(cls.cpu().tolist())
        pclasses.extend(logits.argmax(1).detach().cpu().tolist())
        ids.extend(dish_ids)
        if args.limit and batch > args.limit and split != 'test':
            break
    truths_arr = torch.cat(truths).numpy()
    preds_arr = torch.cat(preds).numpy()
    masks_arr = torch.cat(masks).numpy()
    cls_arr = np.asarray(classes)
    pcls_arr = np.asarray(pclasses)
    return {
        'loss': total / max(n, 1),
        'reg_normalized_l1': reg_total / max(n, 1),
        'n': n,
        'per_field': per_field_metrics(truths_arr, preds_arr, masks_arr),
        'coarse_category_accuracy': float((cls_arr == pcls_arr).mean()) if len(cls_arr) else None,
    }, {'ids': ids, 'truths': truths_arr, 'predictions': preds_arr, 'masks': masks_arr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='每 epoch 批次上限(冒烟)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--init-from', default=None, help='官方 2 目标权重, 用于迁移特征层')
    ap.add_argument('--manifest', default=str(MANIFEST), help='五目标 manifest(默认 meal_macros_v1; P1-C 用 meal_macros_expanded_v1)')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    manifest_path = Path(args.manifest)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    wgt = WEIGHTS_DIR / args.tag
    wgt.mkdir(parents=True, exist_ok=True)
    state = {'pid': os_pid(), 'protocol': 'meal_macros_v1', 'tag': args.tag, 'seed': args.seed}

    def report(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'status.json', state)
        print(json.dumps(values), flush=True)

    set_base_seed(args.seed)
    model = MealMacrosNet(manifest, pretrained=True, input_channels=3).cuda()
    feature_params = list(model.network.features.parameters())
    feature_ids = {id(p) for p in feature_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in feature_ids]
    optimizer = torch.optim.AdamW([{'params': feature_params, 'lr': 1e-5},
                                   {'params': head_params, 'lr': 1e-4}], weight_decay=1e-4)

    if args.init_from:
        src = torch.load(args.init_from, map_location='cpu', weights_only=True)
        sd = src['model']
        # 只复用特征层；分类/回归头新初始化（尺寸不同）
        new_sd = model.state_dict()
        for k, v in sd.items():
            if k.startswith('network.features.') and k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
        report(stage='init_from', features_reused=True)

    if args.smoke:
        seed_epoch(0)
        tr, _ = epoch_pass(model, manifest, 'train', 0, args, optimizer)
        va, _ = epoch_pass(model, manifest, 'val', 0, args)
        atomic_json(out / 'smoke.json', {'passed': True, 'train': tr, 'val': va, 'discarded': True})
        report(stage='smoke_complete', train_reg=tr['reg_normalized_l1'], val_reg=va['reg_normalized_l1'])
        return

    best, best_epoch = float('inf'), 0
    history = []
    begun = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - begun > 8 * 3600:
            report(stage='paused_budget'); return
        seed_epoch(epoch)
        tr, _ = epoch_pass(model, manifest, 'train', epoch, args, optimizer)
        va, _ = epoch_pass(model, manifest, 'val', epoch, args)
        improved = va['reg_normalized_l1'] < best
        if improved:
            best, best_epoch = va['reg_normalized_l1'], epoch
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        payload = {'model': model.state_dict(), 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
                   'history': history, 'config': {'tag': args.tag, 'seed': args.seed, 'epochs': args.epochs,
                                                  'batch': args.batch, 'protocol': 'meal_macros_v1'}}
        if improved:
            torch.save(payload, wgt / 'best.pt')
        torch.save(payload, wgt / 'last.pt')
        atomic_json(out / 'epochs.json', history)
        report(stage='epoch_complete', epoch=epoch, best_epoch=best_epoch,
               reg_norm=va['reg_normalized_l1'], val_kcal_mae=va['per_field']['calories']['mae'])
        if epoch - best_epoch >= 8 or epoch == args.epochs:
            break
    if not history or best_epoch == 0:
        raise RuntimeError('No valid completion')
    best_path = wgt / 'best.pt'
    model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)['model'])
    test_metrics, detail = epoch_pass(model, manifest, 'test', best_epoch, args)
    with (out / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dish_id'] + [f'true_{t}' for t in TARGETS] + [f'pred_{t}' for t in TARGETS])
        for i, d in enumerate(detail['ids']):
            w.writerow([d] + detail['truths'][i].tolist() + detail['predictions'][i].tolist())
    atomic_json(out / 'test_metrics.json', {'best_epoch': best_epoch,
                                            'checkpoint_sha256': file_digest(best_path),
                                            'metrics': test_metrics, 'manifest_sha256': file_digest(manifest_path)})
    report(stage='complete', tag=args.tag, best_epoch=best_epoch, test=test_metrics)


def os_pid():
    import os
    return os.getpid()


if __name__ == '__main__':
    main()
