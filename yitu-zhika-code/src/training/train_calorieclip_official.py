"""CalorieCLIP-style adaptation on the frozen local official-split subset.

Reimplements the pinned public head/config, not the unavailable author training
code. Never loads food-finetuned weights with unknown Nutrition5k test exposure.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest, job_lock
from scripts.prepare_clip_backbone import DEST, MODEL_SHA, AUTHOR_REVISION
from src.training.train_meal_official import MANIFEST, MANIFEST_SHA, seed_epoch, should_flip

OUTPUT = ROOT / 'results/calorieclip_official_v1'
WEIGHTS = ROOT / 'checkpoints/calorieclip_official_v1'
BACKBONE = DEST / 'open_clip_model.safetensors'
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def preprocess():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class CalorieDataset(Dataset):
    def __init__(self, manifest, split, epoch=0):
        self.rows = [r for r in manifest['rows'] if r['split'] == split]
        self.split, self.epoch, self.transform = split, epoch, preprocess()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['image']) as source:
            image = source.convert('RGB')
        if self.split == 'train' and should_flip(row['dish_id'], self.epoch):
            image = ImageOps.mirror(image)
        return self.transform(image), torch.tensor(row['targets'][0], dtype=torch.float32), row['dish_id']


class CalorieCLIP(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # No implicit network downloads; None is for tests/strict checkpoint reload.
        clip = open_clip.create_model('ViT-B-32', pretrained=None, force_quick_gelu=True)
        if pretrained_path is not None:
            clip.load_state_dict(load_file(str(pretrained_path), device='cpu'), strict=True)
        self.visual = clip.visual
        self.visual.requires_grad_(False)
        for block in self.visual.transformer.resblocks[-2:]:
            block.requires_grad_(True)
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, images):
        # Do not use no_grad here: the last two visual blocks must receive gradients.
        # Public reference uses unnormalized CLIP features, not L2 normalization.
        return self.head(self.visual(images)).squeeze(-1)


def scalar_metrics(target, prediction):
    y, p = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('Expected nonempty matching 1D calorie arrays')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Non-finite metric inputs')
    error, nonzero = p-y, y > 0
    variance = float(((y-y.mean())**2).sum())
    mae = float(np.abs(error).mean())
    return {'n': len(y), 'mae': mae, 'rmse': float(np.sqrt((error**2).mean())),
            'r2': float(1-(error**2).sum()/variance) if variance > 0 else None,
            'mape_nonzero_percent': float(np.abs(error[nonzero]/y[nonzero]).mean()*100) if nonzero.any() else None,
            'mape_n': int(nonzero.sum()), 'negative_predictions': int((p < 0).sum()),
            'mae_over_mean_target_percent': mae/float(y.mean())*100 if y.mean() > 0 else None}


def epoch_pass(model, manifest, split, epoch, report, optimizer=None, max_batches=None):
    training = optimizer is not None
    if training != (split == 'train'):
        raise ValueError('Only training rows can update weights')
    model.train(training)
    loader = DataLoader(CalorieDataset(manifest, split, epoch), batch_size=16,
        shuffle=training, num_workers=0, pin_memory=True, drop_last=False,
        generator=torch.Generator().manual_seed(42+epoch))
    truth, predictions, ids_out, total = [], [], [], 0.0
    for batch, (images, target, ids) in enumerate(loader, 1):
        check_stop(OUTPUT)
        if max_batches is not None and batch > max_batches:
            break
        images, target = images.cuda(), target.cuda()
        if not torch.isfinite(images).all() or not torch.isfinite(target).all():
            raise FloatingPointError('Non-finite batch')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(images)
            loss = nn.functional.huber_loss(prediction.float(), target.float(), delta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite Huber loss')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        truth.extend(target.detach().cpu().tolist())
        predictions.extend(prediction.detach().float().cpu().tolist())
        ids_out.extend(ids)
        total += loss.item()*len(ids)
        if batch == 1 or batch % 25 == 0:
            report(stage='training' if training else split, epoch=epoch,
                   batch=batch, batches=len(loader), loss=total/len(truth))
    return {'huber_loss': total/len(truth), **scalar_metrics(truth, predictions)}, \
        {'ids': ids_out, 'truth': truth, 'predictions': predictions}


def verify_inputs():
    if file_digest(MANIFEST) != MANIFEST_SHA or file_digest(BACKBONE) != MODEL_SHA:
        raise ValueError('Frozen manifest or generic CLIP hash mismatch')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['target_names'] != ['calories', 'mass'] or manifest['target_units'] != ['kcal', 'g']:
        raise ValueError('Calorie target contract mismatch')
    return manifest


def validate_cpu():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = verify_inputs()
    seed_epoch(0)
    model = CalorieCLIP(BACKBONE).train()
    dataset = CalorieDataset(manifest, 'train')
    examples = [dataset[i] for i in range(2)]
    prediction = model(torch.stack([e[0] for e in examples]))
    loss = nn.functional.huber_loss(prediction.float(), torch.stack([e[1] for e in examples]))
    loss.backward()
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        expected = name.startswith(('head.', 'visual.transformer.resblocks.10.', 'visual.transformer.resblocks.11.'))
        if parameter.requires_grad != expected:
            raise AssertionError('Wrong trainability: '+name)
        if expected:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError('Missing or non-finite gradient: '+name)
            trainable.append(name)
        else:
            if parameter.grad is not None:
                raise AssertionError('Frozen parameter received gradient: '+name)
            frozen.append(name)
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise AssertionError('Non-finite CPU validation')
    audit = {'passed': True, 'manifest_sha256': MANIFEST_SHA, 'generic_clip_sha256': MODEL_SHA,
        'script_sha256': file_digest(Path(__file__)), 'training_examples': [e[2] for e in examples],
        'trainable_tensors': len(trainable), 'frozen_tensors': len(frozen),
        'loss': loss.item(), 'discarded_updates': True, 'strict_generic_state_load': True}
    atomic_json(OUTPUT/'cpu_validation.json', audit)
    print(json.dumps(audit), flush=True)


def save(path, payload):
    check_space(ROOT)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'protocol': 'calorieclip_official_v1'}
    def report(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(OUTPUT/'status.json', state)
        print(json.dumps(values), flush=True)
    with job_lock(OUTPUT/'job.lock'):
        try:
            check_stop(OUTPUT)
            check_space(ROOT)
            manifest = verify_inputs()
            if not (ROOT/'results/meal_nir_official_v1/test_metrics.json').exists():
                raise RuntimeError('Finish paired meal run before allocating this GPU job')
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 required, no silent precision fallback')
            cpu_audit = json.loads((OUTPUT/'cpu_validation.json').read_text(encoding='utf-8'))
            if not cpu_audit['passed'] or cpu_audit['script_sha256'] != file_digest(Path(__file__)):
                raise ValueError('Current source needs successful --validate-cpu first')
            config = {'protocol': 'calorieclip_official_v1', 'manifest_sha256': MANIFEST_SHA,
                'generic_clip_sha256': MODEL_SHA, 'author_revision': AUTHOR_REVISION,
                'food_finetuned_weights_used': False, 'target_names': ['calories'], 'target_units': ['kcal'],
                'model': 'ViT-B-32 OpenAI QuickGELU + published 512/256/64 MLP with BN/dropout',
                'trainable_visual': 'last two transformer blocks only', 'feature_l2_normalization': False,
                'epochs': 30, 'batch_size': 16, 'optimizer': 'AdamW',
                'lr_visual': 1e-5, 'lr_head': 1e-3, 'weight_decay': 0.01, 'seed': 42,
                'loss': 'Huber delta=1 on raw kcal', 'early_stopping': False,
                'selection': 'minimum validation calorie MAE', 'amp': 'BF16 with FP32 loss; clip gradient norm 1',
                'preprocessing': {'size': 224, 'resize': 'bicubic shortest edge + center crop', 'mean': MEAN, 'std': STD},
                'augmentation': 'training only dish/epoch hash horizontal flip',
                'local_choices_not_specified_by_author': ['seed', 'weight_decay', 'Huber delta and target scale',
                    'augmentation', 'validation selection', 'gradient clipping', 'BF16'],
                'torch': str(torch.__version__), 'open_clip': str(open_clip.__version__),
                'code_hashes': {str(p.relative_to(ROOT)): file_digest(p) for p in
                    [Path(__file__), ROOT/'src/training/train_meal_official.py']},
                'limitations': ['Public architecture/config adaptation, not exact author-training reproduction.',
                    'Local 2188/567/507 overhead subset; no comparison to author-reported MAE on a different split.',
                    'Single seed; calorie-only output; no weight or category estimates.',
                    'The 128-dish expansion is not included in this matched evaluation.']}
            protocol_path = OUTPUT/'protocol.json'
            if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != config:
                raise ValueError('Frozen protocol changed; use a new version')
            atomic_json(protocol_path, config)
            if (OUTPUT/'test_metrics.json').exists():
                report(stage='complete', already_complete=True)
                return
            budget_path = OUTPUT/'budget.json'
            budget = json.loads(budget_path.read_text(encoding='utf-8')) if budget_path.exists() else {'deadline_unix': time.time()+8*3600}
            atomic_json(budget_path, budget)
            report(stage='verifying_images')
            for row in manifest['rows']:
                check_stop(OUTPUT)
                if file_digest(ROOT/row['image']) != row['image_sha256']:
                    raise ValueError('Image hash mismatch: '+row['dish_id'])
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            def initialize():
                seed_epoch(0)
                model = CalorieCLIP(BACKBONE).cuda()
                optimizer = torch.optim.AdamW([
                    {'params': [p for p in model.visual.parameters() if p.requires_grad], 'lr': 1e-5},
                    {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=0.01)
                return model, optimizer
            model, optimizer = initialize()
            best_path, last_path = WEIGHTS/'best.pt', WEIGHTS/'last.pt'
            start, best, best_epoch, history, finished = 1, float('inf'), 0, [], False
            if last_path.exists():
                last = torch.load(last_path, map_location='cpu', weights_only=True)
                if last['config'] != config:
                    raise ValueError('Resume protocol mismatch')
                model.load_state_dict(last['model'], strict=True)
                optimizer.load_state_dict(last['optimizer'])
                start, best, best_epoch = last['epoch']+1, last['best'], last['best_epoch']
                history, finished = last['history'], last['training_complete']
                del last
            else:
                if best_path.exists():
                    raise ValueError('Best checkpoint without resumable last checkpoint')
                train, _ = epoch_pass(model, manifest, 'train', 0, report, optimizer, max_batches=2)
                val, _ = epoch_pass(model, manifest, 'val', 0, report, max_batches=2)
                atomic_json(OUTPUT/'smoke.json', {'train': train, 'val': val, 'discarded_updates': True})
                del model, optimizer
                torch.cuda.empty_cache()
                model, optimizer = initialize()
            if not finished:
                for epoch in range(start, 31):
                    check_stop(OUTPUT)
                    check_space(ROOT)
                    if time.time() >= budget['deadline_unix']:
                        report(stage='paused_budget')
                        return
                    seed_epoch(epoch)
                    train, _ = epoch_pass(model, manifest, 'train', epoch, report, optimizer)
                    val, _ = epoch_pass(model, manifest, 'val', epoch, report)
                    improved = val['mae'] < best
                    if improved:
                        best, best_epoch = val['mae'], epoch
                    history.append({'epoch': epoch, 'train': train, 'val': val})
                    finished = epoch == 30
                    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'history': history,
                        'config': config, 'training_complete': finished, 'target_names': ['calories']}
                    if improved:
                        save(best_path, checkpoint)
                    save(last_path, checkpoint)
                    atomic_json(OUTPUT/'epochs.json', history)
                    report(stage='epoch_complete', epoch=epoch, best_epoch=best_epoch, validation_calorie_mae=val['mae'])
            if not finished:
                raise RuntimeError('Training incomplete; no test evaluation allowed')
            selected = torch.load(best_path, map_location='cpu', weights_only=True)
            model.load_state_dict(selected['model'], strict=True)
            metrics, details = epoch_pass(model, manifest, 'test', best_epoch, report)
            with (OUTPUT/'test_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['dish_id', 'true_calories', 'pred_calories'])
                writer.writerows(zip(details['ids'], details['truth'], details['predictions']))
            atomic_json(OUTPUT/'test_metrics.json', {'best_epoch': best_epoch, 'completed_epochs': len(history),
                'checkpoint_sha256': file_digest(best_path), 'manifest_sha256': MANIFEST_SHA,
                'predictions_sha256': file_digest(OUTPUT/'test_predictions.csv'), 'metrics': metrics,
                'limitations': config['limitations']})
            report(stage='complete', best_epoch=best_epoch)
        except BaseException as exc:
            report(stage='paused' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-cpu', action='store_true')
    args = parser.parse_args()
    validate_cpu() if args.validate_cpu else run()
