"""Seed-aware Nutrition5k multitask training for S0/S1/S3 ablation.

Wraps the v1 protocol and adds three orthogonal toggles without changing
the manifest split:
  --seed  : all RNGs are reseeded (model init, DataLoader shuffle, flip hash)
  --backbone : optional path to a Food-101-pretrained state_dict
               (keys prefixed 'backbone.' or 'network.backbone.'; head is rebuilt)
  --augment  : enable MixUp + RandAugment + RandomErasing on training only

Outputs:
  results/meal_<tag>_seed<seed>/  {protocol,status,test_metrics,test_predictions,epochs}.json
  checkpoints/meal_<tag>_seed<seed>/{best,last}.pt

The frozen test set is always the same 507 Nutrition5k IDs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import functional as TF
from torchvision.transforms import RandAugment

# ROOT is fixed to the project tree; do not derive from __file__.
ROOT = Path(r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code")
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest, job_lock
from src.models.resnet_multitask import ResNetMultiTask

MANIFEST = ROOT / 'results/meal_official_v1/manifest.json'
MANIFEST_SHA = 'f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa'
TARGET_NAMES = ['calories', 'mass']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _code_hashes(paths):
    """Hash a list of paths; record by relative-to-ROOT if inside ROOT, else by name."""
    out = {}
    for p in paths:
        p = Path(p)
        try:
            key = str(p.relative_to(ROOT))
        except ValueError:
            key = p.name
        out[key] = file_digest(p)
    return out


def seed_all(seed: int, epoch: int):
    s = seed + epoch
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def should_flip(dish_id, epoch, seed):
    return int(hashlib.sha256(f'{seed}:{epoch}:{dish_id}'.encode()).hexdigest()[:8], 16) < 2**31


class MealDataset(Dataset):
    def __init__(self, manifest, split, epoch, seed, augment):
        self.rows = [r for r in manifest['rows'] if r['split'] == split]
        self.split, self.epoch, self.seed, self.augment = split, epoch, seed, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['image']) as source:
            image = source.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
        if self.split == 'train':
            if should_flip(row['dish_id'], self.epoch, self.seed):
                image = ImageOps.mirror(image)
            if self.augment:
                # RandAugment expects PIL input; re-encode for transforms
                image = RandAugment(num_ops=2, magnitude=9)(image)
        rgb = TF.normalize(TF.to_tensor(image), MEAN, STD)
        return rgb, torch.tensor(row['targets'], dtype=torch.float32), row['category_idx'], row['dish_id']


class MealNet(nn.Module):
    def __init__(self, manifest, pretrained=True, backbone_state=None):
        super().__init__()
        self.network = ResNetMultiTask(num_classes=len(manifest['category_to_idx']),
            input_channels=3, pretrained=pretrained, num_regression_targets=2)
        if backbone_state is not None:
            self._load_backbone(backbone_state)
        self.register_buffer('target_center', torch.tensor(manifest['target_stats']['mean'], dtype=torch.float32))
        self.register_buffer('target_scale', torch.tensor(manifest['target_stats']['std'], dtype=torch.float32))

    def _load_backbone(self, sd):
        """Accept state_dicts with prefixes backbone./network.backbone./<none>."""
        # Strip common prefixes
        cleaned = {}
        for k, v in sd.items():
            nk = k
            for p in ('network.backbone.', 'backbone.', 'module.backbone.'):
                if nk.startswith(p):
                    nk = nk[len(p):]
                    break
            cleaned[nk] = v
        own = self.network.state_dict()
        # Only load keys that match shape and aren't the new multitask heads.
        loaded, skipped = 0, []
        for k, v in cleaned.items():
            if k in own and own[k].shape == v.shape and 'fc' not in k:
                own[k] = v
                loaded += 1
            else:
                skipped.append(k)
        self.network.load_state_dict(own)
        print(f"  [backbone] loaded {loaded} tensors; skipped {len(skipped)} (head or shape mismatch)")

    def forward(self, images):
        output = self.network(images)
        return output['logits'], output['nutrition'].float() * self.target_scale + self.target_center


def mixup_batch(images, targets, classes, alpha=0.2):
    if alpha <= 0:
        return images, targets, classes
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, (targets, targets[perm], lam), (classes, classes[perm], lam)


def losses(logits, prediction, target, classes, scale, mix=None, cls_mix=None):
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite regression tensors')
    if mix is None:
        reg = ((prediction - target).abs() / scale).mean()
        cls = nn.functional.cross_entropy(logits.float(), classes)
    else:
        t, t_p, lam = mix
        reg = lam * ((prediction - t).abs() / scale).mean() + (1 - lam) * ((prediction - t_p).abs() / scale).mean()
        c, c_p, _ = cls_mix
        cls = lam * nn.functional.cross_entropy(logits.float(), c) + (1 - lam) * nn.functional.cross_entropy(logits.float(), c_p)
    total = reg + 0.2 * cls
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite loss')
    return total, reg


def regression_metrics(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.ndim != 2 or target.shape != prediction.shape or target.shape[1] != 2:
        raise ValueError('Expected N x 2 targets/predictions')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Non-finite metric inputs')
    err = prediction - target
    mae = np.abs(err).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    ss_res = (err ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    return {'mae_kcal': float(mae[0]), 'mae_g': float(mae[1]),
            'rmse_kcal': float(rmse[0]), 'rmse_g': float(rmse[1]),
            'r2_kcal': float(r2[0]), 'r2_g': float(r2[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='run tag (e.g., s0, s1, s3)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--backbone', type=str, default=None, help='path to pretrained backbone state_dict')
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--mixup', type=float, default=0.0)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--time_budget_s', type=int, default=8 * 3600)
    ap.add_argument('--out_root', type=str, default=None,
                    help='Override base directory for results/ and checkpoints/ (e.g., working dir)')
    args = ap.parse_args()

    seed_all(args.seed, 0)

    base = Path(args.out_root) if args.out_root else ROOT
    OUTPUT = base / 'results' / f'meal_{args.tag}_seed{args.seed}'
    WEIGHTS = base / 'checkpoints' / f'meal_{args.tag}_seed{args.seed}'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    digest = file_digest(MANIFEST)
    if digest != MANIFEST_SHA:
        raise SystemExit('Manifest hash mismatch')

    backbone_sd = None
    if args.backbone:
        ck = torch.load(args.backbone, map_location='cpu', weights_only=True)
        backbone_sd = ck.get('model', ck.get('state_dict', ck))
        print(f"[backbone] loaded {args.backbone}")

    state = {'pid': os.getpid(), 'tag': args.tag, 'seed': args.seed,
             'args': vars(args), 'started_at': datetime.now(timezone.utc).isoformat()}

    def report(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(OUTPUT / 'status.json', state)
        print(json.dumps(values), flush=True)

    with job_lock(OUTPUT / 'job.lock'):
        try:
            check_space(ROOT)
            check_stop(OUTPUT)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 required')

            config = {
                'tag': args.tag, 'seed': args.seed, 'protocol': 'meal_seed_v1',
                'architecture': 'ResNetMultiTask(input=3, classes=11, reg=2, ImageNet backbone)',
                'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
                'backbone': args.backbone, 'augment': args.augment, 'mixup_alpha': args.mixup,
                'patience': args.patience, 'optimizer': 'Adam(2e-4)', 'amp': 'BF16',
                'loss': 'std-normalized L1 + 0.2*CE',
                'selection': 'validation normalized L1', 'patience_early_stop': args.patience,
                'manifest_sha256': digest,
                'code_hashes': _code_hashes([Path(__file__)]),
                'limitations': [
                    'Single seed; multi-seed runs required for significance.',
                    'ImageNet backbone or Food-101-pretrained backbone; not a general vision model.',
                    'Food-101 pretraining itself was single-seed.',
                ],
            }
            protocol = OUTPUT / 'protocol.json'
            if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != config:
                raise SystemExit('Frozen protocol changed')
            atomic_json(protocol, config)

            if (OUTPUT / 'test_metrics.json').exists():
                report(stage='complete', already_complete=True)
                return

            torch.set_num_threads(4)

            def initialize():
                model = MealNet(manifest, pretrained=(backbone_sd is None), backbone_state=backbone_sd).cuda()
                optim = torch.optim.Adam(model.parameters(), lr=args.lr)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5)
                return model, optim, sched

            model, optim, sched = initialize()
            start, best, best_epoch, history, finished = 1, float('inf'), 0, [], False
            last_path, best_path = WEIGHTS / 'last.pt', WEIGHTS / 'best.pt'

            if last_path.exists():
                last = torch.load(last_path, map_location='cpu', weights_only=True)
                if last.get('config') != config:
                    raise SystemExit('Checkpoint config mismatch')
                model.load_state_dict(last['model'])
                optim.load_state_dict(last['optimizer'])
                sched.load_state_dict(last['scheduler'])
                start, best, best_epoch = last['epoch'] + 1, last['best'], last['best_epoch']
                history, finished = last['history'], last['training_complete']
                del last
            else:
                # 2-batch smoke check
                m_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                m_tr = DataLoader(MealDataset(manifest, 'train', 0, args.seed, args.augment),
                                  batch_size=2, shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(args.seed))
                m_va = DataLoader(MealDataset(manifest, 'val', 0, args.seed, False),
                                  batch_size=2, shuffle=False, num_workers=0)
                for rgb, tgt, cls, _ in m_tr:
                    rgb, tgt, cls = rgb.cuda(), tgt.cuda(), cls.cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, pred = model(rgb)
                    l, _ = losses(logits, pred, tgt, cls, model.target_scale)
                    if not torch.isfinite(l):
                        raise FloatingPointError('Smoke non-finite')
                    l.backward(); m_opt.step(); m_opt.zero_grad()
                    break
                for rgb, tgt, cls, _ in m_va:
                    rgb, tgt, cls = rgb.cuda(), tgt.cuda(), cls.cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, pred = model(rgb)
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError('Smoke val non-finite')
                    break
                report(stage='smoke_passed', discarded_updates=True)
                del model, optim, sched
                torch.cuda.empty_cache()
                model, optim, sched = initialize()

            started = time.monotonic()
            for epoch in range(start, args.epochs + 1):
                if time.monotonic() - started > args.time_budget_s:
                    report(stage='paused_budget'); return
                check_space(ROOT); check_stop(OUTPUT)
                seed_all(args.seed, epoch)
                tr_loader = DataLoader(MealDataset(manifest, 'train', epoch, args.seed, args.augment),
                                       batch_size=args.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=True, drop_last=True,
                                       generator=torch.Generator().manual_seed(args.seed + epoch))
                va_loader = DataLoader(MealDataset(manifest, 'val', epoch, args.seed, False),
                                       batch_size=args.batch_size, shuffle=False, num_workers=2,
                                       pin_memory=True)
                # Train
                model.train()
                tr_total, tr_n = 0., 0
                for i, (rgb, tgt, cls, _) in enumerate(tr_loader, 1):
                    rgb, tgt, cls = rgb.cuda(non_blocking=True), tgt.cuda(non_blocking=True), cls.cuda(non_blocking=True)
                    optim.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        if args.mixup > 0:
                            mixed, t_mix, c_mix = mixup_batch(rgb, tgt, cls, args.mixup)
                            logits, pred = model(mixed)
                            loss, reg = losses(logits, pred, tgt, cls, model.target_scale, mix=t_mix, cls_mix=c_mix)
                        else:
                            logits, pred = model(rgb)
                            loss, reg = losses(logits, pred, tgt, cls, model.target_scale)
                    if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optim.step()
                    tr_total += loss.item() * rgb.size(0); tr_n += rgb.size(0)
                tr_loss = tr_total / max(tr_n, 1)
                # Val
                model.eval()
                va_total, va_n = 0., 0
                with torch.no_grad():
                    for rgb, tgt, cls, _ in va_loader:
                        rgb, tgt, cls = rgb.cuda(non_blocking=True), tgt.cuda(non_blocking=True), cls.cuda(non_blocking=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits, pred = model(rgb)
                            loss, _ = losses(logits, pred, tgt, cls, model.target_scale)
                        va_total += loss.item() * rgb.size(0); va_n += rgb.size(0)
                va_loss = va_total / max(va_n, 1)
                sched.step(va_loss)
                improved = va_loss < best
                if improved: best, best_epoch = va_loss, epoch
                history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss,
                                'lr': optim.param_groups[0]['lr']})
                finished = epoch == args.epochs or epoch - best_epoch >= args.patience
                ckpt = {'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optim.state_dict(), 'scheduler': sched.state_dict(),
                        'config': config, 'best': best, 'best_epoch': best_epoch,
                        'history': history, 'training_complete': finished}
                if improved:
                    torch.save(ckpt, best_path)
                torch.save(ckpt, last_path)
                atomic_json(OUTPUT / 'epochs.json', history)
                report(stage='epoch', epoch=epoch, train=tr_loss, val=va_loss,
                       best=best, best_epoch=best_epoch, lr=optim.param_groups[0]['lr'])
                if finished: break

            # Test
            best = torch.load(best_path, map_location='cpu', weights_only=True)
            model.load_state_dict(best['model'])
            te_loader = DataLoader(MealDataset(manifest, 'test', best_epoch, args.seed, False),
                                   batch_size=args.batch_size, shuffle=False, num_workers=2,
                                   pin_memory=True)
            preds, gts, cids, ids = [], [], [], []
            model.eval()
            with torch.no_grad():
                for rgb, tgt, cls, did in te_loader:
                    rgb = rgb.cuda(non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, pred = model(rgb)
                    if not torch.isfinite(pred).all(): raise FloatingPointError('Non-finite test pred')
                    preds.append(pred.float().cpu().numpy())
                    gts.append(tgt.numpy()); cids.append(cls.numpy())
                    ids.extend(did)
            preds = np.concatenate(preds); gts = np.concatenate(gts); cids = np.concatenate(cids)
            # classification accuracy
            # We need logits - re-run with eval
            with torch.no_grad():
                logits_list = []
                for rgb, tgt, cls, _ in te_loader:
                    rgb = rgb.cuda(non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = model(rgb)
                    logits_list.append(logits.float().cpu().numpy())
            logits_arr = np.concatenate(logits_list)
            cls_pred = logits_arr.argmax(1)
            cls_acc = float((cls_pred == cids).mean())
            # CSV
            csv_path = OUTPUT / 'test_predictions.csv'
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['dish_id', 'split', 'cal_true', 'cal_pred', 'mass_true', 'mass_pred', 'class_idx', 'class_pred'])
                for i, did in enumerate(ids):
                    w.writerow([did, 'test', float(gts[i, 0]), float(preds[i, 0]),
                                float(gts[i, 1]), float(preds[i, 1]),
                                int(cids[i]), int(cls_pred[i])])
            metrics = {
                'best_epoch': best_epoch, 'completed_epochs': history[-1]['epoch'],
                'test': {**regression_metrics(gts, preds), 'class_accuracy': cls_acc},
                'checkpoint_sha256': file_digest(best_path),
                'manifest_sha256': digest,
            }
            atomic_json(OUTPUT / 'test_metrics.json', metrics)
            report(stage='complete', best_epoch=best_epoch, test=metrics['test'])
        except BaseException as exc:
            report(stage='paused' if isinstance(exc, InterruptedError) else 'failed',
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
