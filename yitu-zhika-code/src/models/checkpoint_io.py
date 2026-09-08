"""Strict checkpoint loading with explicit physical target names."""
import hashlib
import json
from pathlib import Path

import torch


def resolve_metadata(checkpoint, path, target_count):
    metadata = {k:checkpoint[k] for k in ('target_names', 'classification_valid', 'category_to_idx') if k in checkpoint}
    sidecar = Path(str(path)+'.metadata.json')
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding='utf-8'))
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024**2), b''):
                h.update(chunk)
        if h.hexdigest() != data['checkpoint_sha256']:
            raise ValueError('Checkpoint does not match metadata sidecar hash')
        if metadata.get('target_names') and metadata['target_names'] != data['target_names']:
            raise ValueError('Conflicting checkpoint target names')
        metadata.update(data)
    names = metadata.get('target_names', [])
    if len(names) != target_count or len(set(names)) != len(names):
        raise ValueError('Explicit, unique target_names required; do not infer semantics from head size')
    if not {'calories','mass'}.issubset(names):
        raise ValueError('Checkpoint must explicitly identify calories and mass')
    return metadata


def load_multitask_checkpoint(path, device='cpu'):
    from .resnet_multitask import ResNetMultiTask
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint.get('net_state_dict', checkpoint))
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    targets = state['regressor.3.weight'].shape[0]
    metadata = resolve_metadata(checkpoint, path, targets)
    if 'target_mean' in state:
        raise ValueError('Normalized regression checkpoint requires a separately validated adapter')
    model = ResNetMultiTask(num_classes=state['classifier.1.weight'].shape[0],
                           input_channels=state['features.0.weight'].shape[1],
                           hidden_dim=state['regressor.0.weight'].shape[0],
                           num_regression_targets=targets, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata['epoch'] = checkpoint.get('epoch')
    metadata['classification_valid'] = bool(metadata.get('classification_valid', False))
    return model, metadata


def named_nutrition(values, target_names):
    if len(values) != len(target_names):
        raise ValueError('Prediction/target-name length mismatch')
    return dict(zip(target_names, map(float, values)))
