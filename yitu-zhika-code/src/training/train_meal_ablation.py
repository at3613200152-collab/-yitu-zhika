"""Bounded, resumable, independently switched expanded-dish experiments.

Never promotes weights to Demo. Selection uses validation normalized L1;
the frozen test subset is evaluated only after training finishes.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest, job_lock
from src.models.generator import UNetGenerator
from src.models.meal_ablation import MealAblationNet
from src.training.meal_ablation_core import ARMS, ExpandedDataset, ablation_loss, MEAN, STD
from src.training.train_meal_official import regression_metrics

MANIFEST = ROOT/'results/meal_expanded_v2/manifest.json'
MANIFEST_SHA = '32f4b53665c950fb86c54f842d3d134aaabfcaa5302e15ad7ed82e6898395afb'
GENERATOR = ROOT/'checkpoints/hsi_unet_v2/best.pt'
GENERATOR_SHA = '4157fd4e5837e248dbeef711b22aa515673c412c42894ef98c8523f4de80264b'
SUITE = ROOT/'results/meal_ablation_v2'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save(path, payload):
    check_space(ROOT)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def resume_state(path, config):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['config'] != config:
        raise ValueError('Checkpoint protocol/code/data changed; choose a new version')
    history = state['history']
    if not history:
        raise ValueError('Empty checkpoint history')
    if [h['epoch'] for h in history] != list(range(1, state['epoch']+1)):
        raise ValueError('Incomplete checkpoint history')
    selected = min(history, key=lambda h: h['val']['reg_normalized_l1'])
    if selected['epoch'] != state['best_epoch']:
        raise ValueError('Checkpoint selection does not match validation history')
    # P0-B: 完成状态必须可由「达到预算」或「有效早停」推出，不能只信任布尔值。
    if state.get('training_complete'):
        epochs = config.get('epochs')
        patience = config.get('patience')
        budget_reached = epochs is not None and state['epoch'] >= epochs
        early_stopped = patience is not None and (state['epoch'] - state['best_epoch']) >= patience
        if not (budget_reached or early_stopped):
            raise ValueError('training_complete not supported by budget/early-stopping')
    return state


class FrozenNIR:
    def __init__(self):
        if file_digest(GENERATOR) != GENERATOR_SHA:
            raise ValueError('Frozen generator changed')
        state = torch.load(GENERATOR, map_location='cpu', weights_only=True)
        self.model = UNetGenerator(base_filters=64)
        self.model.load_state_dict(state['G_state_dict'], strict=True)
        self.model.requires_grad_(False).eval().cuda()
        self.mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)

    def __call__(self, images):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.model((images*self.std+self.mean)*2-1).float()
        return ((output+1)/2-.485)/.229


def pass_epoch(model, manifest, split, epoch, seed, settings, batch_size, report,
               out, provider=None, cache=None, optimizer=None, smoke=False,
               gate_override=None, nir_mode='normal', deadline=None):
    training = optimizer is not None
    if training != (split == 'train'):
        raise ValueError('Only train split can update weights')
    model.train(training)
    data = ExpandedDataset(manifest, split, epoch, seed, settings['paired'], cache)
    if smoke:
        if split == 'train':
            # Smoke explicitly exercises real side-view pairs, not only overheads.
            data.rows = ([r for r in data.rows if 'views' in r][:batch_size]
                         +[r for r in data.rows if 'views' not in r][:batch_size])
        else:
            data.rows = data.rows[:batch_size]
    loader = DataLoader(data, batch_size=batch_size, shuffle=training,
                        num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed+epoch))
    n, total, regression, consistency_sum, pair_count = 0, 0., 0., 0., 0
    details = dict(ids=[], truths=[], predictions=[], classes=[], pred_classes=[], gates=[])
    last_gradients = {}

    def forward(rgb, semantics):
        nir = provider(rgb) if provider is not None else None
        if nir is not None and nir_mode == 'shuffled':
            nir = nir.roll(1, dims=0)
        if nir is not None and nir_mode == 'zero':
            nir = torch.zeros_like(nir)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(rgb, nir, semantics, gate_override=gate_override)

    for batch, item in enumerate(loader, 1):
        check_stop(out)
        check_stop(SUITE)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Bounded run deadline reached; resume from last complete epoch')
        rgb = item['rgb'].cuda()
        target, classes = item['targets'].cuda(), item['category'].cuda()
        mask = item['paired'].cuda()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = forward(rgb, item['clip'].cuda())
            second = None
            if training and mask.any():
                second = forward(item['second_rgb'][item['paired']].cuda(),
                                 item['second_clip'][item['paired']].cuda())
                pair_count += int(mask.sum())
            loss, parts = ablation_loss(output, target, classes, model.physical.scale,
                                       model.physical.density_mean, second=second,
                                       pair_mask=mask, consistency_weight=settings['consistency'])
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if smoke:
                    last_gradients = {name: float(p.grad.float().norm()) for name, p in model.named_parameters()
                                      if p.grad is not None and ('gate_net' in name or 'semantic_adapter' in name)}
                optimizer.step()
        size = len(target)
        n += size
        total += loss.item()*size
        regression += parts['reg_normalized_l1'].item()*size
        consistency_sum += parts['consistency'].item()*size
        details['ids'].extend(item['dish_id'])
        details['truths'].extend(target.cpu().tolist())
        details['predictions'].extend(output['nutrition'].detach().float().cpu().tolist())
        details['classes'].extend(classes.cpu().tolist())
        details['pred_classes'].extend(output['logits'].argmax(1).detach().cpu().tolist())
        details['gates'].extend(output['gate'].detach().float().flatten().cpu().tolist())
        if batch == 1 or batch % 50 == 0:
            report(stage='training' if training else split, epoch=epoch, batch=batch, batches=len(loader),
                   loss=total/n, actual_side_pairs=pair_count,
                   gpu_peak_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3))
    if not n:
        raise ValueError('Empty pass')
    metrics = dict(loss=total/n, reg_normalized_l1=regression/n, consistency=consistency_sum/n,
                   n=n, side_pairs=pair_count, gate_mean=sum(details['gates'])/n,
                   metrics=regression_metrics(details['truths'], details['predictions']),
                   coarse_category_accuracy=sum(a == b for a, b in zip(details['classes'], details['pred_classes']))/n)
    if smoke:
        metrics['auxiliary_gradient_norms'] = last_gradients
    return metrics, details


def run(args):
    settings = ARMS[args.arm]
    name = f'{args.arm}_s{args.seed}'
    out = ROOT/('results/meal_ablation_smoke_v2' if args.smoke else 'results/meal_ablation_v2')/name
    weights = ROOT/'checkpoints/meal_ablation_v2'/name
    out.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        weights.mkdir(parents=True, exist_ok=True)
    state = dict(pid=os.getpid(), arm=args.arm, seed=args.seed, smoke=args.smoke)
    started = time.monotonic()
    deadline = started+args.max_hours*3600

    def report(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'status.json', state)
        print(json.dumps(values), flush=True)

    with job_lock(out/'job.lock'):
        try:
            check_stop(out)
            check_stop(SUITE)
            check_space(ROOT)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 is required; no implicit fallback')
            if file_digest(MANIFEST) != MANIFEST_SHA:
                raise ValueError('Expanded manifest changed')
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            cache, cache_audit = None, None
            if settings['clip_aux']:
                from scripts.prepare_clip_semantics import read_cache
                cache, cache_audit = read_cache()
            provider = FrozenNIR() if settings['fusion'] != 'rgb' else None
            image_net = Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
            if not image_net.is_file() or not file_digest(image_net).startswith('11ad3fa6'):
                raise ValueError('Expected verified cached ImageNet initialization')
            code_paths = [Path(__file__), ROOT/'src/training/meal_ablation_core.py',
                          ROOT/'src/models/meal_ablation.py', ROOT/'src/models/resnet_multitask.py',
                          ROOT/'src/models/generator.py', ROOT/'scripts/prepare_clip_semantics.py',
                          ROOT/'src/training/train_meal_official.py']
            config = dict(protocol='expanded_dish_ablation_v2', arm=args.arm, settings=settings,
                          seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, patience=8,
                          manifest_sha256=MANIFEST_SHA, imagenet_sha256=file_digest(image_net),
                          generator_sha256=GENERATOR_SHA if provider else None,
                          clip_cache_sha256=cache_audit['cache_sha256'] if cache_audit else None,
                          target_stats=manifest['target_stats'], density_stats=manifest['density_stats'],
                          category_to_idx=manifest['category_to_idx'], target_names=['calories', 'mass'],
                          optimizer='AdamW; features 1e-5, heads/gate/adapter 1e-4; wd=1e-4',
                          loss='dish-mean normalized L1 + .2 CE; factorized adds .1 normalized density L1; optional .1 paired consistency',
                          selection='minimum validation normalized regression L1; no test tuning',
                          augmentation='one hash-selected view/dish/epoch, matched horizontal flip; optional second distinct same-dish view',
                          preprocessing='256 bilinear resize; ImageNet normalization; frozen NIR generated after flip',
                          precision='CUDA BF16 forward, FP32 physical conversion/loss; gradient clip=1',
                          torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                          code_hashes={str(p.relative_to(ROOT)): file_digest(p) for p in code_paths},
                          limitations=manifest['limitations'])
            protocol = out/'protocol.json'
            if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != config and not args.smoke:
                raise ValueError('Existing protocol differs; use a new experiment version')
            atomic_json(protocol, config)
            if not args.smoke:
                report(stage='verifying_images', counts=manifest['counts'])
                for row in manifest['rows']:
                    for view in row.get('views', [dict(path=row['image'], file_sha256=row['image_sha256'])]):
                        check_stop(out)
                        if file_digest(ROOT/view['path']) != view['file_sha256']:
                            raise ValueError('Image changed: '+row['dish_id'])
            seed_everything(args.seed)
            model = MealAblationNet(manifest, fusion=settings['fusion'], head=settings['head'],
                                    clip_aux=settings['clip_aux']).cuda()
            feature_params = list(model.network.features.parameters())
            feature_ids = {id(p) for p in feature_params}
            head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in feature_ids]
            optimizer = torch.optim.AdamW([dict(params=feature_params, lr=1e-5),
                                          dict(params=head_params, lr=1e-4)], weight_decay=1e-4)

            def epoch_pass(split, epoch, train=False, **kwargs):
                return pass_epoch(model, manifest, split, epoch, args.seed, settings, args.batch_size,
                                  report, out, provider, cache, optimizer if train else None,
                                  smoke=args.smoke, deadline=deadline, **kwargs)

            if args.smoke:
                seed_everything(args.seed)
                train, _ = epoch_pass('train', 0, train=True)
                val, _ = epoch_pass('val', 0)
                if provider and any(p.grad is not None or p.requires_grad for p in provider.model.parameters()):
                    raise ValueError('Frozen generator received gradients')
                # The same real model is forced to ignore changed NIR inputs.
                gate_check = None
                if settings['fusion'] == 'gate':
                    first, _ = epoch_pass('val', 0, gate_override=0.)
                    second, _ = epoch_pass('val', 0, gate_override=0., nir_mode='shuffled')
                    gate_check = first['metrics'] == second['metrics']
                    if not gate_check:
                        raise ValueError('Forced-off gate leaked NIR influence')
                atomic_json(out/'smoke.json', dict(passed=True, train=train, val=val,
                            gate_off_invariance=gate_check, discarded_updates=True, test_used=False))
                report(stage='smoke_complete')
                return
            start, best, best_epoch, history, finished = 1, float('inf'), 0, [], False
            if (weights/'last.pt').exists():
                last = resume_state(weights/'last.pt', config)
                model.load_state_dict(last['model'], strict=True)
                optimizer.load_state_dict(last['optimizer'])
                start, best, best_epoch = last['epoch']+1, last['best'], last['best_epoch']
                history, finished = last['history'], last['training_complete']
                del last
                report(stage='resumed', epoch=start-1)
            elif (weights/'best.pt').exists():
                raise ValueError('Best exists without resumable last checkpoint')
            if not finished:
                for epoch in range(start, args.epochs+1):
                    check_space(ROOT)
                    seed_everything(args.seed+epoch)
                    train, _ = epoch_pass('train', epoch, train=True)
                    val, _ = epoch_pass('val', epoch)
                    improved = val['reg_normalized_l1'] < best
                    if improved:
                        best, best_epoch = val['reg_normalized_l1'], epoch
                    history.append(dict(epoch=epoch, train=train, val=val))
                    finished = epoch == args.epochs or epoch-best_epoch >= 8
                    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   epoch=epoch, best=best, best_epoch=best_epoch, history=history,
                                   config=config, training_complete=finished)
                    if improved:
                        save(weights/'best.pt', dict(model=model.state_dict(), config=config,
                                                   epoch=epoch, validation=val))
                    save(weights/'last.pt', payload)
                    atomic_json(out/'epochs.json', history)
                    report(stage='epoch_complete', epoch=epoch, best_epoch=best_epoch,
                           val_calorie_mae=val['metrics']['calories']['mae'],
                           val_calorie_mape=val['metrics']['calories']['mape_nonzero_percent'])
                    if finished:
                        break
            if not finished:
                raise RuntimeError('No valid completion condition')
            best_state = torch.load(weights/'best.pt', map_location='cpu', weights_only=True)
            if best_state['config'] != config or best_state['epoch'] != best_epoch:
                raise ValueError('Best checkpoint provenance mismatch')
            model.load_state_dict(best_state['model'], strict=True)
            if not (out/'completion_audit.json').exists():
                metrics, details = epoch_pass('test', best_epoch)
                expected = {r['dish_id']: r for r in manifest['rows'] if r['split'] == 'test'}
                if set(details['ids']) != expected.keys() or len(details['ids']) != len(expected):
                    raise ValueError('Test dishes mismatch')
                for key, target in zip(details['ids'], details['truths']):
                    if not np.array_equal(np.asarray(target), np.asarray(expected[key]['targets'], dtype=np.float32)):
                        raise ValueError('Test target drift')
                path = out/'test_predictions.csv'
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['dish_id','true_calories','true_mass','pred_calories','pred_mass',
                                     'true_coarse_class','pred_coarse_class'])
                    for i, key in enumerate(details['ids']):
                        writer.writerow([key, *details['truths'][i], *details['predictions'][i],
                                         details['classes'][i], details['pred_classes'][i]])
                result = dict(best_epoch=best_epoch, manifest_sha256=MANIFEST_SHA,
                              checkpoint_sha256=file_digest(weights/'best.pt'), metrics=metrics,
                              limitations=config['limitations'])
                atomic_json(out/'test_metrics.json', result)
                if settings['fusion'] == 'gate':
                    probes = {}
                    for label, kwargs in [('off', dict(gate_override=0.)), ('on', dict(gate_override=1.)),
                                          ('shuffled', dict(nir_mode='shuffled'))]:
                        probes[label], _ = epoch_pass('val', best_epoch, **kwargs)
                    atomic_json(out/'validation_gate_probes.json', dict(probes=probes,
                                checkpoint_selected_before_probes=True, used_for_selection=False))
                atomic_json(out/'completion_audit.json', dict(passed=True, samples=len(expected),
                            checkpoint_sha256=result['checkpoint_sha256'], predictions_sha256=file_digest(path),
                            manifest_sha256=MANIFEST_SHA, best_epoch=best_epoch,
                            scope='Runner checked fixed test IDs/targets and validation selection; independent audit still required'))
            report(stage='complete', best_epoch=best_epoch)
        except BaseException as exc:
            paused = isinstance(exc, (TimeoutError, InterruptedError))
            report(stage='paused_budget_or_stop' if paused else 'failed', error=f'{type(exc).__name__}: {exc}')
            if paused:
                return
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43], default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-hours', type=float, default=8.)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 60 or not 2 <= args.batch_size <= 16 or not 0 < args.max_hours <= 12:
        parser.error('Invalid bounded training budget')
    run(args)
