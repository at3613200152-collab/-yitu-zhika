"""阶段一复现：在全量 HSIFoodIngr-64（2772 训练 / 290 验证 / 327 测试）上训练 RGB→NIR 生成器。

与本地 144 扫描版（hsi_unet_v2）保持同一协议，便于对照，也便于与论文 PSNR/SSIM 讨论：
- 结构：UNetGenerator（4 层下采样，base 64，输入 3 通道 → 输出 1 通道）
- 目标：860 nm 单波段，训练集统计缩放后的 [0,1]（见 data/hsi_full_v3/manifest.json 的 conversion 字段）
- 损伤：L1（[-1,1] 域）；Adam 2e-4 β(.5,.999)；ReduceLROnPlateau 0.5/5；bf16；梯度裁剪 1.0
- 选模：验证集逐图平均 L1（[0,1] 域），patience 10
- 指标：逐图 PSNR `10log10(1/MSE)`、SSIM（range=1, gaussian 11×11），另报"批 MSE 口径"
- 溯源：清单/代码 SHA、标定参数、旋转方式、划分计数、采样规则全部写入权重 config

用法：
  python src/training/train_hsi_full_v3.py --tag full_seed42 --epochs 30 --batch 8
  python src/training/train_hsi_full_v3.py --tag full_smoke --epochs 1 --limit 3   # 冒烟
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest, check_stop   # noqa: E402
from src.models.generator import UNetGenerator                          # noqa: E402

DATA = ROOT / 'data/hsi_full_v3'
MANIFEST = DATA / 'manifest.json'
OUTPUT = ROOT / 'results/hsi_full_v3'
WEIGHTS = ROOT / 'checkpoints/hsi_full_v3'


class Pairs(torch.utils.data.Dataset):
    def __init__(self, manifest, split, epoch=0, seed=42):
        self.rows = [r for r in manifest['rows'] if r['split'] == split]
        self.paths = manifest['_paths'][split]
        self.rgb = np.load(self.paths['rgb'], mmap_mode='r')
        self.nir = np.load(self.paths['nir'], mmap_mode='r')
        self.split, self.epoch, self.seed = split, epoch, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb = np.asarray(self.rgb[row['index']]).astype(np.float32) / 255.0
        nir = np.asarray(self.nir[row['index']]).astype(np.float32)
        if self.split == 'train':
            h = int(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["id"]}'.encode()).hexdigest()[:8], 16)
            if h < 2 ** 31:
                rgb, nir = rgb[:, ::-1].copy(), nir[:, ::-1].copy()
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))) * 2 - 1
        nir_t = torch.from_numpy(np.ascontiguousarray(nir))[None] * 2 - 1
        return rgb_t, nir_t, row['id']


def metrics(pred01, tgt01):
    mse = float(np.mean((pred01 - tgt01) ** 2))
    return {'l1_01': float(np.mean(np.abs(pred01 - tgt01))),
            'psnr_db': 10 * np.log10(1.0 / max(mse, 1e-12)), 'mse': mse,
            'ssim': float(sk_ssim(tgt01, pred01, data_range=1.0, gaussian_weights=True, sigma=1.5))}


def pass_epoch(model, manifest, split, epoch, batch, report, optimizer=None, limit=None, seed=42):
    train = optimizer is not None
    loader = torch.utils.data.DataLoader(Pairs(manifest, split, epoch, seed), batch_size=batch,
                                         shuffle=train, num_workers=0, pin_memory=True,
                                         generator=torch.Generator().manual_seed(seed + epoch))
    total, n, rows = 0.0, 0, []
    for i, (rgb, nir, ids) in enumerate(loader, 1):
        check_stop(OUTPUT)
        if limit and i > limit and split != 'test':
            break
        rgb, nir = rgb.cuda(), nir.cuda()
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(rgb)
            loss = nn.functional.l1_loss(pred.float(), nir)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite generator loss')
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        n += len(ids)
        total += loss.item() * len(ids)
        if not train:
            p = ((pred.float() + 1) / 2).clamp(0, 1).detach().cpu().numpy()[:, 0]
            t = ((nir + 1) / 2).clamp(0, 1).cpu().numpy()[:, 0]
            for key, a, b in zip(ids, p, t):
                rows.append(dict(id=key, **metrics(a, b)))
        if i % 50 == 0 or i == 1:
            report(stage='training' if train else split, epoch=epoch, batch=i, batches=len(loader),
                   loss=total / max(n, 1))
    out = {'loss': total / max(n, 1), 'n': n}
    if rows:
        out.update({k: float(np.mean([r[k] for r in rows])) for k in ('l1_01', 'psnr_db', 'ssim')})
    return out, rows


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_seed42')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='每 epoch 最大 batch 数（冒烟）')
    ap.add_argument('--max-hours', type=float, default=7.5)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    manifest['_paths'] = {s: {'rgb': str(DATA / f'rgb_{s}.npy'), 'nir': str(DATA / f'nir_{s}.npy')}
                          for s in ('train', 'val', 'test')}
    out = OUTPUT / args.tag
    wgt = WEIGHTS / args.tag

    def nonempty(p):
        return p.exists() and any(p.iterdir())
    if nonempty(out) or nonempty(wgt):
        raise SystemExit(f'[拒绝覆盖] 输出目录已存在且非空: {out} / {wgt}；请换 --tag')
    out.mkdir(parents=True, exist_ok=True)
    wgt.mkdir(parents=True, exist_ok=True)

    state = {'protocol': 'hsi_full_v3', 'tag': args.tag, 'seed': args.seed}

    def report(**v):
        state.update(v, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(out / 'status.json', state)
        print(json.dumps(v), flush=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = UNetGenerator(in_channels=3, out_channels=1, base_filters=64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    config = {'protocol': 'hsi_full_v3', 'tag': args.tag, 'seed': args.seed, 'epochs': args.epochs,
              'batch_size': args.batch, 'architecture': 'UNetGenerator(4-down, base=64)',
              'loss': 'L1 on [-1,1]', 'optimizer': 'Adam 2e-4 beta(.5,.999)',
              'scheduler': 'ReduceLROnPlateau 0.5/5', 'amp': 'bf16 forward, fp32 loss, clip 1.0',
              'selection': 'val mean per-image L1 ([0,1])', 'patience': 10,
              'manifest': str(MANIFEST.relative_to(ROOT)), 'manifest_sha256': file_digest(MANIFEST),
              'code_sha256': file_digest(Path(__file__)), 'counts': manifest['counts'],
              'conversion': manifest['conversion'], 'sampler_seed_base': args.seed,
              'sampler_rule': 'flip by sha256(seed:epoch:id); DataLoader generator manual_seed(seed+epoch)'}

    print(f"训练/验证/测试: {manifest['counts']}", flush=True)
    best, best_epoch, history = float('inf'), 0, []
    began = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - began > args.max_hours * 3600:
            report(stage='paused_budget', epoch=epoch - 1, best_epoch=best_epoch)
            break
        tr, _ = pass_epoch(model, manifest, 'train', epoch, args.batch, report, optimizer, args.limit, args.seed)
        va, _ = pass_epoch(model, manifest, 'val', epoch, args.batch, report, None, args.limit, args.seed)
        scheduler.step(va['l1_01'])
        improved = va['l1_01'] < best
        if improved:
            best, best_epoch = va['l1_01'], epoch
        history.append({'epoch': epoch, 'train': tr, 'val': va,
                        'lr': optimizer.param_groups[0]['lr']})
        payload = {'model': model.state_dict(), 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
                   'config': config}
        save(wgt / 'last.pt', payload)
        if improved:
            save(wgt / 'best.pt', payload)
        atomic_json(out / 'epochs.json', history)
        atomic_json(out / 'config.json', config)
        report(stage='epoch_complete', epoch=epoch, best_epoch=best_epoch, val_l1=va['l1_01'],
               val_psnr=va['psnr_db'], val_ssim=va['ssim'])
        if epoch - best_epoch >= 10:
            report(stage='early_stop', epoch=epoch, best_epoch=best_epoch)
            break

    model.load_state_dict(torch.load(wgt / 'best.pt', map_location='cpu', weights_only=True)['model'])
    test, rows = pass_epoch(model, manifest, 'test', best_epoch, args.batch, report, None, None, args.seed)
    with (out / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'l1_01', 'psnr_db', 'mse', 'ssim'])
        for r in rows:
            w.writerow([r['id'], r['l1_01'], r['psnr_db'], r['mse'], r['ssim']])
    batch_mse = float(np.mean([r['mse'] for r in rows]))
    summary = {'protocol': 'hsi_full_v3', 'tag': args.tag, 'best_epoch': best_epoch,
               'val_l1_best': best, 'test': test, 'n_test': len(rows),
               'test_psnr_batch_mse_db': 10 * np.log10(1.0 / max(batch_mse, 1e-12)),
               'checkpoint_sha256': file_digest(wgt / 'best.pt'),
               'config': config}
    atomic_json(out / 'test_metrics.json', summary)
    report(stage='complete', tag=args.tag, best_epoch=best_epoch, test_psnr=test['psnr_db'],
           test_ssim=test['ssim'], test_l1=test['l1_01'])
    print(json.dumps({'tag': args.tag, 'best_epoch': best_epoch, 'n_test': len(rows),
                      'test_l1_01': round(test['l1_01'], 5), 'test_psnr_db': round(test['psnr_db'], 3),
                      'test_psnr_batch_mse_db': round(summary['test_psnr_batch_mse_db'], 3),
                      'test_ssim': round(test['ssim'], 4),
                      'checkpoint_sha256': summary['checkpoint_sha256'][:16]}), flush=True)


if __name__ == '__main__':
    main()
