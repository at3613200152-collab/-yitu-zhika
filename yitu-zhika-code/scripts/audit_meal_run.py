"""Independent artifact audit; no model inference, optimization, or test tuning."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.capsicum_job import atomic_json,file_digest


def run():
    output = ROOT/'results/meal_rgb_official_v1'
    weights = ROOT/'checkpoints/meal_rgb_official_v1'
    manifest_path = ROOT/'results/meal_official_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = json.loads((output/'test_metrics.json').read_text(encoding='utf-8'))
    history = json.loads((output/'epochs.json').read_text(encoding='utf-8'))
    best = torch.load(weights/'best.pt',map_location='cpu',weights_only=True)
    last = torch.load(weights/'last.pt',map_location='cpu',weights_only=True)
    assert file_digest(weights/'best.pt')==result['checkpoint_sha256']
    assert file_digest(manifest_path)==result['manifest_sha256']
    assert last['training_complete'] and last['epoch']==30 and len(history)==30
    assert history==last['history']
    selected = min(history,key=lambda e:e['val']['reg_normalized_l1'])['epoch']
    assert best['epoch']==selected==result['best_epoch']==last['best_epoch']
    assert best['target_names']==last['target_names']==manifest['target_names']==['calories','mass']
    assert manifest['target_units']==['kcal','g']
    assert best['config']==last['config']==json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    assert best['config']['manifest_sha256']==result['manifest_sha256']
    for checkpoint in (best,last):
        assert all(torch.isfinite(v).all().item() for v in checkpoint['model'].values())
    with (output/'test_predictions.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    by_id = {r['dish_id']:r for r in manifest['rows'] if r['split']=='test'}
    assert len(rows)==len(by_id)==len({r['dish_id'] for r in rows})==507
    assert {r['dish_id'] for r in rows}==set(by_id)
    metrics = result['metrics']['metrics']
    recomputed = {}
    for index,name in enumerate(['calories','mass']):
        y = np.array([float(r['true_'+name]) for r in rows])
        p = np.array([float(r['pred_'+name]) for r in rows])
        reference = np.array([by_id[r['dish_id']]['targets'][index] for r in rows],dtype=np.float32).astype(np.float64)
        np.testing.assert_array_equal(y,reference)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        error = p-y
        nonzero = y>0
        actual = {'n':len(y),'mae':float(np.abs(error).mean()),
            'rmse':float(np.sqrt(np.mean(error**2))),
            'r2':float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),
            'mape_nonzero_percent':float(np.mean(np.abs(error[nonzero]/y[nonzero]))*100),
            'mape_n':int(nonzero.sum()),
            'mae_over_mean_target_percent':float(np.abs(error).mean()/y.mean()*100),
            'negative_predictions':int((p<0).sum())}
        for key,value in actual.items():
            np.testing.assert_allclose(value,metrics[name][key],rtol=1e-10,atol=1e-10)
        recomputed[name] = actual
    assert all(int(r['true_coarse_class'])==by_id[r['dish_id']]['category_idx'] for r in rows)
    category_accuracy = sum(int(r['true_coarse_class'])==int(r['pred_coarse_class']) for r in rows)/len(rows)
    np.testing.assert_allclose(category_accuracy,result['metrics']['coarse_category_accuracy'])
    training_values = np.array([r['targets'] for r in manifest['rows'] if r['split']=='train'])
    np.testing.assert_allclose(training_values.mean(0),manifest['target_stats']['mean'])
    np.testing.assert_allclose(training_values.std(0),manifest['target_stats']['std'])
    pointer = json.loads((ROOT/'results/video_expansion_pilot.json').read_text(encoding='utf-8'))
    assert file_digest(pointer['manifest'])==pointer['sha256'] and pointer['included_in_training'] is False
    pilot = json.loads(Path(pointer['manifest']).read_text(encoding='utf-8'))
    assert len(pilot['rows'])==4 and pilot['complete'] and sum(len(r['frames']) for r in pilot['rows'])==12
    pilot_ids = {r['dish_id'] for r in pilot['rows']}
    assert not pilot_ids & {r['dish_id'] for r in manifest['rows']}
    audit = {'passed':True,'completed_epochs':30,'selected_epoch':selected,'test_rows':len(rows),
        'checkpoint_sha256':result['checkpoint_sha256'],'manifest_sha256':result['manifest_sha256'],
        'predictions_sha256':file_digest(output/'test_predictions.csv'),'recomputed_metrics':recomputed,
        'coarse_category_accuracy':category_accuracy,'pilot_separate':True,
        'method':'CPU checkpoint/hash/CSV recomputation only; no additional model inference or selection',
        'limitations':'Verifies saved artifact consistency, not original capture independence or scientific significance.'}
    atomic_json(output/'completion_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
