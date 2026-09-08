"""Reproducible auxiliary Capsicum RGB->NIR experiment (not calorie evidence).

Four-level U-Net from scratch, L1 supervision, paired flips, official splits.
Validation chooses checkpoints; the test split is evaluated only at the end.
"""
import argparse
from importlib.metadata import version
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from models.generator import UNetGenerator
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest

NUMERIC_POLICY = 'amp_finite_loss_bounded_skip_v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Pairs(Dataset):
    def __init__(self, manifest, split, img_size=256):
        self.root = Path(manifest['root'])
        self.rows = [r for r in manifest['pairs'] if r['split'] == split]
        self.split = split
        self.img_size = img_size
        if not self.rows:
            raise ValueError(f'Empty {split} split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tensors = []
        flip = self.split == 'train' and random.random() < 0.5
        for field in ('rgb', 'nir'):
            with Image.open(self.root / row[field]) as image:
                image = image.convert('RGB' if field == 'rgb' else 'L')
                image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                if flip:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            if field == 'rgb':
                array = array.transpose(2, 0, 1)
            else:
                array = array[None]
            tensors.append(torch.from_numpy(array.copy()))
        return tuple(tensors)


def train_epoch(model, loader, optimizer, scaler, device, progress, max_batches=None):
    model.train()
    total, count = 0.0, 0
    skipped, consecutive, successful = 0, 0, 0
    for index, (rgb, nir) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        rgb, nir = rgb.to(device), nir.to(device)
        if not torch.isfinite(rgb).all() or not torch.isfinite(nir).all():
            raise FloatingPointError('Nonfinite training input or target')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            pred = model(rgb)
            loss = nn.functional.l1_loss(pred, nir)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            raise RuntimeError('No gradients produced')
        if not torch.stack([torch.isfinite(g).all() for g in grads]).all():
            if not scaler.is_enabled():
                raise FloatingPointError('Nonfinite gradients without AMP scaling')
            skipped += 1
            consecutive += 1
            if consecutive > 3 or skipped > 8:
                raise FloatingPointError('Persistent AMP overflow; bounded recovery exhausted')
            old_scale = scaler.get_scale()
            # unscale_ already detected bad gradients: step skips optimizer mutation,
            # update lowers the scale. Do not clip inf/NaN gradients into the model.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            progress(event='amp_overflow', batch=index+1, total_batches=len(loader),
                     previous_scale=old_scale, scale=scaler.get_scale(),
                     skipped_steps=skipped, successful_steps=successful)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        consecutive = 0
        successful += 1
        total += loss.item() * rgb.size(0)
        count += rgb.size(0)
        if index % 20 == 0:
            progress(event='train_batch', batch=index+1, total_batches=len(loader), train_l1=loss.item(),
                     skipped_steps=skipped, successful_steps=successful, scale=scaler.get_scale())
    if not count:
        raise FloatingPointError('No successful training steps; no finite update was performed')
    progress(event='train_epoch_end', skipped_steps=skipped, successful_steps=successful,
             scale=scaler.get_scale())
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    sums = {'l1': 0.0, 'psnr_db': 0.0, 'ssim': 0.0}
    count = 0
    for index, (rgb, nir) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        rgb, nir = rgb.to(device), nir.to(device)
        pred = model(rgb)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite validation output')
        pred01 = ((pred.float() + 1) / 2).clamp(0, 1)
        target01 = (nir.float() + 1) / 2
        mse = (pred01-target01).square().flatten(1).mean(1)
        l1 = (pred-nir).abs().flatten(1).mean(1)
        sums['l1'] += l1.sum().item()
        sums['psnr_db'] += (-10 * torch.log10(mse.clamp_min(1e-12))).sum().item()
        for p, t in zip(pred01[:, 0].cpu().numpy(), target01[:, 0].cpu().numpy()):
            sums['ssim'] += float(structural_similarity(t, p, data_range=1.0))
        count += rgb.size(0)
    if not count:
        raise ValueError('Empty evaluation loader')
    return {**{k: v/count for k, v in sums.items()}, 'samples': count}


def save_weights(path, payload):
    check_space(path.parent)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def make_loaders(manifest, batch_size):
    return {split: DataLoader(Pairs(manifest, split), batch_size=batch_size,
                              shuffle=split=='train', num_workers=0,
                              pin_memory=True, drop_last=split=='train')
            for split in ('train', 'val')}


def smoke(manifest, batch_size, device):
    seed_all(42)
    loaders = make_loaders(manifest, batch_size)
    model = UNetGenerator(base_filters=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    loss = train_epoch(model, loaders['train'], optimizer, scaler, device,
                       lambda **kw: print('SMOKE', kw, flush=True), max_batches=2)
    metrics = evaluate(model, loaders['val'], device, max_batches=2)
    del optimizer, scaler, model, loaders
    torch.cuda.empty_cache()
    return {'train_l1': loss, 'validation': metrics, 'max_batches_each': 2}


def synthetic_smoke():
    """Exercise the real GPU train/eval path with generated fixtures, not results."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently start a CPU overnight run')
    with tempfile.TemporaryDirectory(prefix='capsicum-gpu-smoke-') as d:
        root = Path(d)
        rows = []
        rng = np.random.default_rng(42)
        for split in ('train', 'val'):
            for i in range(16):
                arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                rgb, nir = f'{split}_{i}_rgb.png', f'{split}_{i}_nir.png'
                Image.fromarray(arr).save(root/rgb)
                Image.fromarray(arr[:,:,0]).save(root/nir)
                rows.append({'split': split, 'rgb': rgb, 'nir': nir})
        result = smoke({'root': str(root), 'pairs': rows}, 8, torch.device('cuda'))
        print(json.dumps({'synthetic_gpu_test_only': result}), flush=True)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing automatic CPU fallback')
    device = torch.device('cuda')
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_hash = file_digest(manifest_path)
    run_name = 'capsicum_unet4_seed42'
    result_dir = ROOT / 'results' / run_name
    ckpt_dir = ROOT / 'checkpoints' / run_name
    for folder in (result_dir, ckpt_dir):
        folder.mkdir(parents=True, exist_ok=True)
    config = {'architecture': 'UNetGenerator4', 'base_filters': 64, 'loss': 'L1',
              'lr': 2e-4, 'seed': 42, 'img_size': 256, 'batch_size': args.batch_size,
              'epochs': args.epochs, 'early_stopping_patience': 15,
              'max_training_hours': 8, 'manifest_sha256': manifest_hash,
              'initialization': 'from_scratch', 'selection': 'validation_l1',
              'augmentation': 'paired horizontal flip only', 'amp': True,
              'numeric_policy': NUMERIC_POLICY,
              'environment': {'python': sys.version, 'torch': torch.__version__,
                              'cuda': torch.version.cuda, 'numpy': np.__version__,
                              'scikit_image': version('scikit-image')},
              'psnr': 'per-image [0,1] MSE; mean dB; MSE floor 1e-12',
              'ssim': 'skimage structural_similarity, data_range=1, mean per-image',
              'limitations': manifest.get('limitations', '')}
    status = {'pid': os.getpid(), 'device': torch.cuda.get_device_name(0), 'config': config}
    def progress(**values):
        check_stop(manifest_path.parent)
        check_space(ckpt_dir)
        status.update(values)
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(result_dir / 'training_status.json', status)
        print(json.dumps(values, ensure_ascii=False), flush=True)
    try:
        metrics_path = result_dir / 'test_metrics.json'
        if metrics_path.exists():
            previous = json.loads(metrics_path.read_text(encoding='utf-8'))
            saved_config = json.loads((result_dir/'protocol.json').read_text(encoding='utf-8'))
            if saved_config != config or previous['manifest_sha256'] != manifest_hash:
                raise ValueError('Completed experiment differs; use a new run directory')
            progress(stage='complete', test_metrics=str(metrics_path), already_complete=True)
            return
        progress(stage='smoke')
        smoke_metrics = smoke(manifest, args.batch_size, device)
        atomic_json(result_dir / 'smoke.json', smoke_metrics)
        progress(stage='smoke_passed', smoke=smoke_metrics)
        seed_all(42)  # Discard smoke updates; formal run starts independently.
        loaders = make_loaders(manifest, args.batch_size)
        model = UNetGenerator(base_filters=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        scaler = torch.amp.GradScaler('cuda')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        first_epoch, best, stale = 1, float('inf'), 0
        last_path = ckpt_dir / 'last_model.pth'
        if last_path.exists():
            last = torch.load(last_path, map_location='cpu', weights_only=False)
            if last['config'] != config:
                previous_config = {k: v for k, v in config.items() if k != 'numeric_policy'}
                if not getattr(args, 'allow_amp_policy_migration', False) or last['config'] != previous_config:
                    raise ValueError('Existing experiment config differs; choose a new run directory or explicitly migrate only the verified AMP policy')
                backup = ckpt_dir / 'before_amp_fix'
                backup.mkdir(exist_ok=True)
                for name in ('last_model.pth', 'best_model.pth'):
                    source, destination = ckpt_dir/name, backup/name
                    if destination.exists() and file_digest(destination) != file_digest(source):
                        raise ValueError('Existing pre-fix backup differs; refusing overwrite')
                    if not destination.exists():
                        shutil.copy2(source, destination)
                atomic_json(result_dir/'amp_policy_migration.json', {
                    'timestamp':datetime.now(timezone.utc).isoformat(), 'resume_after_epoch':last['epoch'],
                    'source_sha256':file_digest(backup/'last_model.pth'), 'backup_directory':str(backup),
                    'old_config':last['config'], 'new_config':config,
                    'evidence':'results/debug_capsicum/replay_probe.json',
                    'note':'Only finite-loss AMP gradient overflow handling changes; data, model, optimizer and RNG are resumed unchanged.'})
            model.load_state_dict(last['G_state_dict'], strict=True)
            optimizer.load_state_dict(last['optimizer'])
            scheduler.load_state_dict(last['scheduler'])
            scaler.load_state_dict(last['scaler'])
            first_epoch, best, stale = last['epoch'] + 1, last['best_l1'], last['stale']
            torch.set_rng_state(last['torch_rng'])
            torch.cuda.set_rng_state_all(last['cuda_rng'])
            random.setstate(last['python_rng'])
            np.random.set_state(last['numpy_rng'])
            del last
        atomic_json(result_dir / 'protocol.json', config)
        started = time.monotonic()
        stop_reason = 'max_epochs'
        for epoch in range(first_epoch, args.epochs+1):
            progress(stage='training', epoch=epoch, epochs=args.epochs)
            loss = train_epoch(model, loaders['train'], optimizer, scaler, device, progress)
            val = evaluate(model, loaders['val'], device)
            scheduler.step(val['l1'])
            improved = val['l1'] < best
            stale = 0 if improved else stale+1
            if improved:
                best = val['l1']
                save_weights(ckpt_dir / 'best_model.pth',
                             {'epoch': epoch, 'G_state_dict': model.state_dict(),
                              'config': config, 'validation': val})
            save_weights(last_path, {'epoch': epoch, 'G_state_dict': model.state_dict(),
                         'config': config, 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                         'best_l1': best, 'stale': stale, 'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                         'numpy_rng': np.random.get_state()})
            record = {'epoch': epoch, 'train_l1': loss, 'validation': val,
                      'best_l1': best, 'lr': optimizer.param_groups[0]['lr'],
                      'skipped_steps':status.get('skipped_steps', 0),
                      'successful_steps':status.get('successful_steps', 0),
                      'amp_scale':scaler.get_scale(), 'numeric_policy':NUMERIC_POLICY}
            with (result_dir/'epochs.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            progress(stage='epoch_saved', **record)
            if stale >= 15 or time.monotonic()-started >= 8*3600:
                stop_reason = 'early_stopping' if stale >= 15 else 'time_budget'
                break
        # Do not repeatedly evaluate test on a rerun of a completed experiment.
        metrics_path = result_dir / 'test_metrics.json'
        if not metrics_path.exists():
            progress(stage='final_test')
            best_ckpt = torch.load(ckpt_dir/'best_model.pth', map_location='cpu', weights_only=False)
            model.load_state_dict(best_ckpt['G_state_dict'], strict=True)
            test_loader = DataLoader(Pairs(manifest, 'test'), batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, pin_memory=True)
            test = evaluate(model, test_loader, device)
            atomic_json(metrics_path, {'best_epoch': best_ckpt['epoch'], 'test': test,
                                      'checkpoint_sha256': file_digest(ckpt_dir/'best_model.pth'),
                                      'manifest_sha256': manifest_hash, 'stop_reason': stop_reason,
                                      'scope': 'Capsicum auxiliary NIR only; not food-calorie accuracy'})
        progress(stage='complete', stop_reason=stop_reason, test_metrics=str(metrics_path))
    except BaseException as exc:
        status.update(stage='failed', error=f'{type(exc).__name__}: {exc}')
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(result_dir / 'training_status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--synthetic-self-test', action='store_true')
    parser.add_argument('--allow-amp-policy-migration', action='store_true')
    args = parser.parse_args()
    if args.synthetic_self_test:
        synthetic_smoke()
    elif not args.manifest:
        parser.error('--manifest is required for a real experiment')
    elif args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be >=1 and batch-size >=2')
    else:
        run(args)
