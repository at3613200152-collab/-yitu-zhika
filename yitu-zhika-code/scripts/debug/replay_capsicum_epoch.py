"""Replay saved epoch RNG/state without modifying production checkpoints."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.train_capsicum import make_loaders, train_epoch, UNetGenerator
from scripts.capsicum_job import atomic_json, file_digest


def replay(max_batches, probe=False, output='replay_fixed.json'):
    path = ROOT / 'checkpoints/capsicum_unet4_seed42/last_model.pth'
    manifest_path = ROOT / 'data/deepNIR_capsicum/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['config']['manifest_sha256'] == file_digest(manifest_path)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loaders = make_loaders(manifest, checkpoint['config']['batch_size'])
    model = UNetGenerator(base_filters=64).to(device)
    model.load_state_dict(checkpoint['G_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler = torch.amp.GradScaler('cuda')
    scaler.load_state_dict(checkpoint['scaler'])
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    random.setstate(checkpoint['python_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    record = {'source_checkpoint_sha256': file_digest(path), 'saved_epoch': checkpoint['epoch'],
              'initial_scaler': scaler.state_dict(), 'production_files_unchanged': True}
    del checkpoint
    capture = {}
    original_loss = torch.nn.functional.l1_loss
    original_clip = torch.nn.utils.clip_grad_norm_
    def before_forward(module, inputs):
        capture['rgb'] = inputs[0].detach().clone()
        capture['cuda_rng'] = torch.cuda.get_rng_state_all()
        capture['buffers'] = {n: b.clone() for n, b in model.named_buffers()}
    def loss_capture(prediction, target, *a, **kw):
        capture['nir'] = target.detach().clone()
        return original_loss(prediction, target, *a, **kw)
    def clip_capture(parameters, *a, **kw):
        parameters = list(parameters)
        bad = sum(not torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None)
        if bad:
            capture['bad_grad_tensors'] = bad
            capture['state'] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            capture['state'].update({n: b.cpu().clone() for n, b in capture['buffers'].items()})
            capture['input_finite'] = bool(torch.isfinite(capture['rgb']).all())
            capture['target_finite'] = bool(torch.isfinite(capture['nir']).all())
            capture['weights_finite'] = all(torch.isfinite(p).all().item() for p in model.parameters())
            capture['optimizer_finite'] = all(torch.isfinite(v).all().item() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
        return original_clip(parameters, *a, **kw)
    hook = model.register_forward_pre_hook(before_forward) if probe else None
    def progress(**values):
        record.update(values)
        print('[DEBUG-capsicum-replay]', json.dumps(values), flush=True)
    start = time.monotonic()
    try:
        with patch('torch.nn.functional.l1_loss', loss_capture if probe else original_loss), patch('torch.nn.utils.clip_grad_norm_', clip_capture if probe else original_clip):
            loss = train_epoch(model, loaders['train'], optimizer, scaler, device, progress, max_batches=max_batches)
        record.update(outcome='completed_without_failure', train_l1=loss)
    except Exception as exc:
        record.update(outcome='failure_reproduced', exception=type(exc).__name__, error=str(exc))
    record.update(final_scaler=scaler.state_dict(), elapsed_seconds=time.monotonic()-start)
    if hook:
        hook.remove()
    if probe and 'state' in capture:
        record['failure_checks'] = {k: capture[k] for k in ('bad_grad_tensors', 'input_finite', 'target_finite', 'weights_finite', 'optimizer_finite')}
        record['precision_probes'] = []
        for amp, scale in [(True, 131072.0), (True, 65536.0), (True, 1.0), (False, 1.0)]:
            model.load_state_dict(capture['state'], strict=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.set_rng_state_all(capture['cuda_rng'])
            with torch.autocast('cuda', enabled=amp):
                loss = original_loss(model(capture['rgb']), capture['nir'])
            (loss * scale).backward()
            finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            record['precision_probes'].append({'amp':amp, 'scale':scale, 'loss':loss.item(), 'gradients_finite':finite})
        fixture_path = ROOT / 'results/debug_capsicum/failing_batch.pt'
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'G_state_dict':capture['state'], 'rgb':capture['rgb'].cpu(), 'nir':capture['nir'].cpu(), 'cuda_rng':capture['cuda_rng']}, fixture_path)
        record['failing_batch_fixture'] = str(fixture_path)
    atomic_json(ROOT / 'results/debug_capsicum' / ('replay_probe.json' if probe else output), record)
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-batches', type=int, default=161)
    parser.add_argument('--probe-overflow', action='store_true')
    parser.add_argument('--output', default='replay_fixed.json')
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        parser.error('--output must be a filename')
    replay(args.max_batches, args.probe_overflow, args.output)
