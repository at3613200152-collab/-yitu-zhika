"""Pretrain ResNet50 on Food-101; save only the backbone (drop 101-class head).

The output is consumed by train_meal_seed.py --backbone <path>.
Food-101 test split is used here for backbone validation only and is NEVER
mixed into Nutrition5k evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import functional as TF
from torchvision.transforms import RandAugment

# ROOT is fixed to the project tree; do not derive from __file__ (this script
# may be invoked from a different working directory).
ROOT = Path(r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code")
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest, job_lock

MANIFEST = ROOT / 'data/food101_manifest/manifest.json'
OUTPUT = ROOT / 'checkpoints/food101_pretrained'
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


def seed_all(seed, epoch=0):
    s = seed + epoch
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Food101(Dataset):
    def __init__(self, rows, augment):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row['image']) as img:
            img = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        if self.augment:
            img = RandAugment(num_ops=2, magnitude=9)(img)
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        return x, row['class_idx']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--time_budget_s', type=int, default=8 * 3600)
    args = ap.parse_args()

    seed_all(args.seed, 0)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    train_rows, test_rows = manifest['rows_train'], manifest['rows_test']
    print(f"[INFO] train={len(train_rows)} test={len(test_rows)} classes={manifest['num_classes']}")

    state = {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(), 'args': vars(args)}

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
                'protocol': 'food101_resnet50_pretrain_v1',
                'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
                'optimizer': 'Adam(1e-3)', 'loss': 'CE(label_smoothing=0.1)',
                'amp': 'BF16', 'backbone': 'ResNet50(ImageNet V2) -> 101 classes',
                'selection': 'test top-1 accuracy',
                'manifest_sha256': file_digest(MANIFEST),
                'code_hashes': _code_hashes([Path(__file__)]),
                'limitations': [
                    'Food-101 train labels are noisy; ~10-20% of training images mislabeled.',
                    'Pretrained on the official train split; test split used only for accuracy, never for Nutrition5k.',
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
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 101)
            model = model.cuda()
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
            scaler = None  # using torch.autocast for BF16
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

            start, best_acc, best_epoch, history, finished = 1, 0., 0, [], False
            last_path, best_path = OUTPUT / 'last.pt', OUTPUT / 'best.pt'
            if last_path.exists():
                last = torch.load(last_path, map_location='cpu', weights_only=True)
                if last['config'] != config:
                    raise SystemExit('Checkpoint config mismatch')
                model.load_state_dict(last['model'])
                optim.load_state_dict(last['optimizer'])
                sched.load_state_dict(last['scheduler'])
                start, best_acc, best_epoch = last['epoch'] + 1, last['best_acc'], last['best_epoch']
                history, finished = last['history'], last['training_complete']
                del last
            else:
                # 1-batch smoke
                tr_loader = DataLoader(Food101(train_rows[:args.batch_size * 2], augment=False),
                                       batch_size=4, num_workers=0)
                for x, y in tr_loader:
                    x, y = x.cuda(), y.cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = model(x)
                    l = loss_fn(out, y)
                    if not torch.isfinite(l): raise FloatingPointError('Smoke non-finite')
                    l.backward(); optim.zero_grad()
                    break
                report(stage='smoke_passed')
                del model, optim, sched
                torch.cuda.empty_cache()
                model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, 101)
                model = model.cuda()
                optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

            started = time.monotonic()
            for epoch in range(start, args.epochs + 1):
                if time.monotonic() - started > args.time_budget_s:
                    report(stage='paused_budget'); return
                check_space(ROOT); check_stop(OUTPUT)
                seed_all(args.seed, epoch)
                tr_loader = DataLoader(Food101(train_rows, augment=True),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
                                       generator=torch.Generator().manual_seed(args.seed + epoch))
                te_loader = DataLoader(Food101(test_rows, augment=False),
                                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True, persistent_workers=args.num_workers > 0)
                # Train
                model.train()
                tr_loss, tr_correct, tr_n = 0., 0, 0
                for x, y in tr_loader:
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    optim.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = model(x)
                        loss = loss_fn(out, y)
                    if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optim.step()
                    tr_loss += loss.item() * x.size(0)
                    tr_correct += (out.argmax(1) == y).float().sum().item()
                    tr_n += x.size(0)
                tr_loss /= tr_n; tr_acc = tr_correct / tr_n
                # Test
                model.eval()
                te_correct, te_n = 0, 0
                with torch.no_grad():
                    for x, y in te_loader:
                        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            out = model(x)
                        te_correct += (out.argmax(1) == y).float().sum().item()
                        te_n += x.size(0)
                te_acc = te_correct / te_n
                sched.step()
                improved = te_acc > best_acc
                if improved: best_acc, best_epoch = te_acc, epoch
                history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
                                'test_acc': te_acc, 'lr': optim.param_groups[0]['lr']})
                finished = epoch == args.epochs
                ckpt = {'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optim.state_dict(), 'scheduler': sched.state_dict(),
                        'config': config, 'best_acc': best_acc, 'best_epoch': best_epoch,
                        'history': history, 'training_complete': finished}
                if improved:
                    torch.save(ckpt, best_path)
                torch.save(ckpt, last_path)
                atomic_json(OUTPUT / 'epochs.json', history)
                report(stage='epoch', epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                       test_acc=te_acc, best_acc=best_acc, best_epoch=best_epoch,
                       lr=optim.param_groups[0]['lr'])
                if finished: break

            # Final: save backbone only
            best = torch.load(best_path, map_location='cpu', weights_only=True)
            model.load_state_dict(best['model'])
            # Drop fc; keep everything else
            backbone = {k: v for k, v in model.state_dict().items() if not k.startswith('fc.')}
            bb_path = OUTPUT / 'backbone_only.pt'
            torch.save({'model': backbone, 'config': config,
                        'best_epoch': best['best_epoch'], 'best_test_acc': best['best_acc'],
                        'manifest_sha256': file_digest(MANIFEST)}, bb_path)
            metrics = {'best_epoch': best['best_epoch'],
                       'best_test_acc': best['best_acc'],
                       'backbone_sha256': file_digest(bb_path),
                       'full_ckpt_sha256': file_digest(best_path),
                       'manifest_sha256': file_digest(MANIFEST)}
            atomic_json(OUTPUT / 'test_metrics.json', metrics)
            report(stage='complete', **metrics)
        except BaseException as exc:
            report(stage='paused' if isinstance(exc, InterruptedError) else 'failed',
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
