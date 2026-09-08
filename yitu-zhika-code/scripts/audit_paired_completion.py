"""Audit saved paired-model results and separately downloaded expansion, offline."""
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest
from src.data.meal_manifest import capture_day, training_partition


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_models():
    torch.set_num_threads(2)
    manifest_path = ROOT/'results/meal_official_v1/manifest.json'
    manifest = read(manifest_path)
    test = {r['dish_id']: r for r in manifest['rows'] if r['split'] == 'test'}
    ids = sorted(test)
    predictions, summaries = {}, {}
    for name in ('meal_rgb_official_v1', 'meal_nir_official_v1'):
        folder, weights = ROOT/'results'/name, ROOT/'checkpoints'/name
        result, protocol, history = [read(folder/f'{part}.json') for part in ('test_metrics', 'protocol', 'epochs')]
        best = torch.load(weights/'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(weights/'last.pt', map_location='cpu', weights_only=True)
        assert result['manifest_sha256'] == file_digest(manifest_path) == protocol['manifest_sha256']
        assert result['checkpoint_sha256'] == file_digest(weights/'best.pt')
        assert last['training_complete'] and history == last['history']
        assert last['epoch'] == len(history) and [e['epoch'] for e in history] == list(range(1, len(history)+1))
        assert last['epoch'] == 30 or last['epoch']-last['best_epoch'] >= 8
        selected = min(history, key=lambda e: e['val']['reg_normalized_l1'])['epoch']
        assert selected == best['epoch'] == last['best_epoch'] == result['best_epoch']
        assert best['config'] == last['config'] == protocol
        assert best['target_names'] == manifest['target_names'] == ['calories', 'mass']
        assert manifest['target_units'] == ['kcal', 'g']
        for checkpoint in (best, last):
            assert all(torch.isfinite(v).all().item() for v in checkpoint['model'].values())
        if name == 'meal_nir_official_v1':
            generator = torch.load(ROOT/'checkpoints/hsi_unet_v2/best.pt', map_location='cpu', weights_only=True)
            assert file_digest(ROOT/'checkpoints/hsi_unet_v2/best.pt') == result['generator_sha256'] == protocol['generator_sha256']
            for key, value in generator['G_state_dict'].items():
                assert torch.equal(value, best['model']['generator.'+key])
            del generator
        with (folder/'test_predictions.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        by_id = {r['dish_id']: r for r in rows}
        assert len(rows) == len(by_id) == len(test) == 507 and set(by_id) == set(test)
        y = np.asarray([test[d]['targets'] for d in ids], dtype=np.float32).astype(np.float64)
        recorded = np.asarray([[float(by_id[d]['true_'+k]) for k in ('calories', 'mass')] for d in ids])
        np.testing.assert_array_equal(y, recorded)
        p = np.asarray([[float(by_id[d]['pred_'+k]) for k in ('calories', 'mass')] for d in ids])
        assert np.isfinite(p).all()
        for i, target in enumerate(('calories', 'mass')):
            errors = p[:, i]-y[:, i]
            nonzero = y[:, i] > 0
            actual = {'mae': float(np.abs(errors).mean()), 'rmse': float(np.sqrt((errors**2).mean())),
                'r2': float(1-(errors**2).sum()/((y[:, i]-y[:, i].mean())**2).sum()),
                'mape_nonzero_percent': float(np.abs(errors[nonzero]/y[nonzero, i]).mean()*100),
                'mape_n': int(nonzero.sum()), 'negative_predictions': int((p[:, i] < 0).sum())}
            for key, value in actual.items():
                np.testing.assert_allclose(value, result['metrics']['metrics'][target][key], atol=1e-10, rtol=1e-10)
        assert all(int(by_id[d]['true_coarse_class']) == test[d]['category_idx'] for d in ids)
        accuracy = sum(int(by_id[d]['pred_coarse_class']) == test[d]['category_idx'] for d in ids)/len(ids)
        np.testing.assert_allclose(accuracy, result['metrics']['coarse_category_accuracy'])
        summaries[name] = {'passed': True, 'selected_epoch': selected, 'completed_epochs': len(history),
            'checkpoint_sha256': result['checkpoint_sha256'], 'manifest_sha256': result['manifest_sha256'],
            'predictions_sha256': file_digest(folder/'test_predictions.csv'), 'metrics': result['metrics']}
        predictions[name] = p
        # Preserve the more extensive historical RGB audit; save paired-run audits separately.
        atomic_json(folder/'paired_completion_audit.json', summaries[name])
        del best, last
    difference = np.abs(predictions['meal_nir_official_v1']-y)-np.abs(predictions['meal_rgb_official_v1']-y)
    days = np.asarray([test[d]['capture_day_utc'] for d in ids])
    unique_days = sorted(set(days))
    sums = np.asarray([difference[days == day].sum(0) for day in unique_days])
    counts = np.asarray([(days == day).sum() for day in unique_days])
    rng = np.random.default_rng(20260906)
    draws = rng.integers(len(unique_days), size=(10000, len(unique_days)))
    bootstrap = sums[draws].sum(1)/counts[draws].sum(1)[:, None]
    result = {'passed': True, 'test_dishes': len(ids), 'test_capture_days': len(unique_days),
        'models': summaries, 'delta_direction': 'RGB+predicted-NIR minus RGB; negative MAE delta favors NIR',
        'mae_delta': dict(zip(['calories', 'mass'], difference.mean(0).tolist())),
        'capture_day_bootstrap_95_percentile_interval': dict(zip(['calories', 'mass'], np.quantile(bootstrap, [.025, .975], axis=0).T.tolist())),
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20260906,
        'limitations': ['Conditional on these two fixed single-seed models; not training-seed uncertainty.',
            'Capture days are only proxies for dependence, not verified physical plate groups.',
            'Small MAE difference is not sufficient evidence of a stable NIR benefit; MAPE worsens.']}
    atomic_json(ROOT/'results/paired_comparison_v1.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}, indent=2), flush=True)


def audit_expansion():
    pointer = read(ROOT/'results/video_expansion_v1.json')
    expansion = read(pointer['manifest'])
    assert file_digest(pointer['manifest']) == pointer['sha256'] and pointer['included_in_training'] is False
    assert expansion['complete'] and len(expansion['rows']) == 128
    plan_path = Path(pointer['manifest']).parent/'download_plan.json'
    assert file_digest(plan_path) == pointer['plan_sha256'] == expansion['plan_sha256']
    plan = {r['dish_id']: r for r in read(plan_path)['rows']}
    overhead = read(ROOT/'results/meal_official_v1/manifest.json')
    official_train = set((ROOT/'data/Nutrition5k/official_metadata/rgb_train_ids.txt').read_text().splitlines())
    official_test = set((ROOT/'data/Nutrition5k/official_metadata/rgb_test_ids.txt').read_text().splitlines())
    with (ROOT/'data/Nutrition5k/dishes_verified.csv').open(encoding='utf-8', newline='') as stream:
        labels = {r['dish_id']: r for r in csv.DictReader(stream)}
    assert file_digest(ROOT/'data/Nutrition5k/dishes_verified.csv') == expansion['sources']['metadata_sha256']
    ids, pixels, short = set(), set(), []
    for row in expansion['rows']:
        dish = row['dish_id']
        assert dish not in ids and dish in official_train and dish not in official_test
        assert row['split'] == training_partition(dish) == 'train'
        assert row['capture_day_utc'] == capture_day(dish)
        assert row['targets'] == [float(labels[dish]['total_calories']), float(labels[dish]['total_mass'])]
        assert row['object'] == plan[dish]['object'] and row['targets'] == plan[dish]['targets']
        video = Path(row['video'])
        assert video.stat().st_size == int(row['object']['size'])
        assert file_digest(video, 'md5') == row['verified_md5'] == base64.b64decode(row['object']['md5Hash']).hex()
        assert len(row['frames']) == 3
        if [f['index'] for f in row['frames']] != [0, 30, 60]:
            short.append(dish)
        for frame in row['frames']:
            assert file_digest(frame['path']) == frame['file_sha256']
            with Image.open(frame['path']) as image:
                image = image.convert('RGB')
                assert list(image.size) == frame['size']
                pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            assert pixel_sha == frame['pixel_sha256'] and pixel_sha not in pixels
            pixels.add(pixel_sha)
        ids.add(dish)
    assert not ids & {r['dish_id'] for r in overhead['rows']}
    assert not pixels & {r['pixel_sha256'] for r in overhead['rows']}
    result = {'passed': True, 'dishes': len(ids), 'frames': len(pixels),
        'manifest_sha256': pointer['sha256'], 'planned_video_bytes': pointer['planned_video_bytes'],
        'short_video_dishes': short, 'included_in_training': False,
        'checks': 'All raw video MD5, PNG SHA256, decoded pixels, labels, official train IDs, day partition, frozen plan, no exact overhead overlap.',
        'limitations': 'Frames from one dish are correlated; 384 frames are not 384 independent dishes. Near-duplicates are not ruled out.'}
    atomic_json(ROOT/'results/video_expansion_v1_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    audit_models()
    audit_expansion()
