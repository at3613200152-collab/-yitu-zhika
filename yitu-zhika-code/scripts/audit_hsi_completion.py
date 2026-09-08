"""Validate HSI experiment artifacts and render fixed illustrative test predictions."""
import json
from pathlib import Path
import sys
import torch
torch.set_num_threads(2)
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.capsicum_job import atomic_json,file_digest
from src.models.generator import UNetGenerator


def run():
    output=ROOT/'results/hsi_unet_v2'
    manifest_path=ROOT/'data/hsi_prepared_v2/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    result=json.loads((output/'test_metrics.json').read_text(encoding='utf-8'))
    history=json.loads((output/'epochs.json').read_text(encoding='utf-8'))
    checkpoint_path=ROOT/'checkpoints/hsi_unet_v2/best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert file_digest(checkpoint_path)==result['checkpoint_sha256']
    assert file_digest(manifest_path)==result['manifest_sha256']==checkpoint['config']['manifest_sha256']
    selected=min(history,key=lambda r:r['val']['l1_minus1_plus1'])['epoch']
    assert checkpoint['epoch']==result['best_epoch']==selected
    assert result['completed_epochs']==len(history)==100
    rows=json.loads((output/'test_predictions.json').read_text(encoding='utf-8'))
    tests=[r for r in manifest['rows'] if r['split']=='test']
    assert len(rows)==len(tests)==len({r['id'] for r in rows})==33
    assert {r['id'] for r in rows}=={r['id'] for r in tests}
    for key in ('l1_01','psnr_db','ssim'):
        np.testing.assert_allclose(np.mean([r[key] for r in rows]),result['test'][key],rtol=1e-7)
    assert all(torch.isfinite(t).all() for t in checkpoint['G_state_dict'].values())
    clipping={s:{key:float(np.mean([r[key] for r in manifest['rows'] if r['split']==s]))
                 for key in ('nir_below_range_fraction','nir_above_range_fraction')}
              for s in ('train','val','test')}
    atomic_json(output/'completion_audit.json',{'passed':True,'best_epoch':selected,'completed_epochs':len(history),
        'test_samples':len(rows),'checkpoint_sha256':result['checkpoint_sha256'],
        'manifest_sha256':result['manifest_sha256'],'mean_native_clipping_fraction':clipping})
    torch.set_num_threads(2)
    model=UNetGenerator(base_filters=64)
    model.load_state_dict(checkpoint['G_state_dict'],strict=True)
    model.eval()
    examples=[]
    for i,row in enumerate(tests[:3]):
        pair=np.load(ROOT/row['prepared'],allow_pickle=False)
        with torch.no_grad():
            pred=((model(torch.from_numpy(pair[:3][None])*2-1)[0,0]+1)/2).numpy()
        examples.append(np.concatenate([pair,pred[None]],axis=0))
    np.savez_compressed(output/'test_examples.npz',images=np.stack(examples),ids=np.asarray([r['id'] for r in tests[:3]]))
    # Plotting uses an isolated NumPy process, avoiding mixed OpenMP libraries.
    import subprocess
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/plot_hsi_examples.py')],check=True)
    print(json.dumps({'audit':'passed','test':result['test'],'best_epoch':selected,'completed_epochs':len(history),'clipping':clipping}),flush=True)


if __name__=='__main__':
    run()
