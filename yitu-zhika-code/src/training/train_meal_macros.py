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


def _atomic_save(payload, path):
    """原子保存：先写 .tmp 再替换，避免中断留下半写文件。"""
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


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
                        generator=torch.Generator().manual_seed(args.seed + epoch))
    total, reg_total, n = 0.0, 0.0, 0
    truths, preds, masks, classes, pclasses, ids = [], [], [], [], [], []
    for batch, (images, target, mask, cls, dish_ids) in enumerate(loader, 1):
        images, target, mask, cls = images.cuda(), target.cuda(), mask.cuda(), cls.cuda()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, prediction = model(images)
            loss, reg = macros_losses(logits, prediction, target, cls, model.target_scale, mask,
                                      class_weight=getattr(args, 'class_weight_tensor', None))
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
    }, {'ids': ids, 'truths': truths_arr, 'predictions': preds_arr, 'masks': masks_arr,
        'classes': cls_arr, 'pred_classes': pcls_arr}


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
    ap.add_argument('--resume', action='store_true', help='显式续跑: 从 last.pt 恢复(需要 --tag 已有未完成产物)')
    ap.add_argument('--nonneg-output', action='store_true',
                    help='方案 §7.1 B：五目标用 softplus 参数化（输出恒非负）')
    ap.add_argument('--class-weight-ce', action='store_true',
                    help='方案 §7.1 C：分类头用逐类权重 CE（频率平方根倒数、封顶 4.0）')
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    manifest_path = Path(args.manifest)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / args.tag
    wgt = WEIGHTS_DIR / args.tag

    # 防覆盖（整改清单 §6.2）：非空输出目录必须显式 --resume；否则拒绝，不用同命令静默覆盖
    def _nonempty(p):
        return p.exists() and any(p.iterdir())
    if not args.smoke and not args.resume and (_nonempty(out) or _nonempty(wgt)):
        raise SystemExit(
            f'[拒绝覆盖] 输出目录已存在且非空: {out} / {wgt}。'
            f'请使用新的 --tag，或确认要续跑时显式加 --resume。')

    out.mkdir(parents=True, exist_ok=True)
    wgt.mkdir(parents=True, exist_ok=True)
    state = {'pid': os_pid(), 'protocol': 'meal_macros_v1', 'tag': args.tag, 'seed': args.seed}

    def report(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'status.json', state)
        print(json.dumps(values), flush=True)

    set_base_seed(args.seed)
    model = MealMacrosNet(manifest, pretrained=True, input_channels=3,
                          nonneg=args.nonneg_output).cuda()
    # 方案 §7.1 C：逐类权重 CE（权重来自 train 划分，先定后看，不按测试集调参）
    class_weight = None
    if args.class_weight_ce:
        from src.training.meal_ablation_core import class_weights_from_manifest
        class_weight, weight_info = class_weights_from_manifest(manifest, device='cuda')
        args.class_weight_tensor = class_weight
        print(json.dumps({'stage': 'class_weight', **weight_info}), flush=True)
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
    start_epoch = 1
    if args.resume:
        last_path = wgt / 'last.pt'
        if not last_path.exists():
            raise SystemExit(f'[--resume] 缺少 {last_path}，无法续跑（请新建 --tag）')
        # last.pt 含 optimizer/numpy 随机状态，weights_only=True 无法反序列化；
        # 这是本机训练自己写出的受信文件，故显式 weights_only=False。
        st = torch.load(last_path, map_location='cpu', weights_only=False)
        if st.get('config', {}).get('protocol') not in (None, 'meal_macros_v1'):
            raise SystemExit('[--resume] last.pt 协议不匹配，拒绝续跑；请新建 --tag')
        if st.get('config', {}).get('manifest_sha256') and st['config']['manifest_sha256'] != file_digest(manifest_path):
            raise SystemExit('[--resume] manifest 已变化，拒绝在旧产物上续跑；请新建 --tag')
        model.load_state_dict(st['model'], strict=True)
        if 'optimizer' in st:
            optimizer.load_state_dict(st['optimizer'])
        start_epoch = int(st['epoch']) + 1
        best, best_epoch = st.get('best', float('inf')), st.get('best_epoch', 0)
        history = list(st.get('history', []))
        rng = st.get('rng')
        if rng:
            import random as _r
            _r.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
        report(stage='resumed', next_epoch=start_epoch, epochs_planned=args.epochs)

    begun = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        if time.monotonic() - begun > 8 * 3600:
            report(stage='paused_budget'); return
        seed_epoch(epoch)
        tr, _ = epoch_pass(model, manifest, 'train', epoch, args, optimizer)
        va, _ = epoch_pass(model, manifest, 'val', epoch, args)
        improved = va['reg_normalized_l1'] < best
        if improved:
            best, best_epoch = va['reg_normalized_l1'], epoch
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        # 两种产物分离（§6.2）：
        # - best.pt 必须能用 weights_only=True 加载（推理侧/评估侧只读它），因此只放张量+标量；
        # - last.pt 是续跑产物，含 optimizer/numpy 随机状态（weights_only 不允许的对象），
        #   由 --resume 以 weights_only=False 读取本地受信文件。
        import random as _rng
        config = {'tag': args.tag, 'seed': args.seed, 'epochs': args.epochs,
                  'batch': args.batch, 'protocol': 'meal_macros_v1',
                  'num_classes': len(manifest['category_to_idx']),
                  'label_schema_version': manifest.get('label_schema_version', 'v1_11class'),
                  'sampler_seed_base': args.seed,
                  'nonneg_output': bool(args.nonneg_output),
                  'class_weight_ce': bool(args.class_weight_ce),
                  'sampler_rule': 'seed_epoch(epoch) 全局种子 + DataLoader 生成器 manual_seed(seed+epoch)',
                  'manifest': str(manifest_path),
                  'manifest_sha256': file_digest(manifest_path),
                  'code_sha256': file_digest(Path(__file__))}
        payload = {'model': model.state_dict(),
                   'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
                   'config': config}
        if improved:
            _atomic_save(payload, wgt / 'best.pt')
        resume_payload = dict(payload, optimizer=optimizer.state_dict(), history=history,
                              rng={'python': _rng.getstate(), 'numpy': np.random.get_state(),
                                   'torch': torch.get_rng_state()})
        _atomic_save(resume_payload, wgt / 'last.pt')
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
    # 逐盘预测需带类别列，否则无法用 scripts/audit_classification_support.py 做逐类支持度审计
    idx_to_name = {v: k for k, v in manifest['category_to_idx'].items()}
    with (out / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dish_id'] + [f'true_{t}' for t in TARGETS] + [f'pred_{t}' for t in TARGETS]
                   + [f'mask_{t}' for t in TARGETS]
                   + ['true_coarse_class', 'pred_coarse_class', 'true_coarse_name', 'pred_coarse_name'])
        for i, d in enumerate(detail['ids']):
            tcls, pcls = int(detail['classes'][i]), int(detail['pred_classes'][i])
            w.writerow([d] + detail['truths'][i].tolist() + detail['predictions'][i].tolist()
                       + detail['masks'][i].astype(int).tolist()
                       + [tcls, pcls, idx_to_name.get(tcls), idx_to_name.get(pcls)])
    atomic_json(out / 'test_metrics.json', {'best_epoch': best_epoch,
                                            'checkpoint_sha256': file_digest(best_path),
                                            'metrics': test_metrics, 'manifest_sha256': file_digest(manifest_path),
                                            'label_schema_version': manifest.get('label_schema_version', 'v1_11class'),
                                            'num_classes': len(manifest['category_to_idx'])})
    report(stage='complete', tag=args.tag, best_epoch=best_epoch, test=test_metrics)


def os_pid():
    import os
    return os.getpid()


if __name__ == '__main__':
    main()
