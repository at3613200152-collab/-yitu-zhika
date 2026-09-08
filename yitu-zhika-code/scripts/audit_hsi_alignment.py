"""Check native PNG/cube orientation using training dates only; never auto-register."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.filters import sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.hsi_manifest import parse_header,read_nir
from scripts.capsicum_job import atomic_json,file_digest


def run(path,rotation,all_training):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    selected = {}
    for row in manifest['rows']:
        if row['split']=='train':
            selected.setdefault(row['id'] if all_training else row['capture_day'],row)
    results = []
    for day,row in selected.items():
        hdr = parse_header(ROOT/row['hdr'])
        visible,_,nm = read_nir(ROOT/row['raw'],hdr,550.)
        rgb = np.asarray(Image.open(ROOT/row['rgb_source']).convert('RGB'),dtype=np.float32)/255.
        a = sobel(rgb.mean(2))
        b = sobel(visible)
        scores = {}
        for rotation in range(4):
            for mirror in (False,True):
                candidate = np.rot90(b,rotation)
                if mirror:
                    candidate = np.fliplr(candidate)
                scores[f'rot{rotation*90}_mirror{int(mirror)}'] = float(np.corrcoef(a.ravel(),candidate.ravel())[0,1])
        identity = scores[f'rot{rotation*90}_mirror0']
        passed = np.isfinite(list(scores.values())).all() and identity>=max(scores.values())-.01 and identity>.5
        results.append({'id':row['id'],'day':row['capture_day'],'visible_nm':nm,'scores':scores,'passed':bool(passed)})
    audit = {'manifest_sha256':file_digest(path),'training_scans_checked':len(results),'nir_rot90_k':rotation,
             'passed':all(r['passed'] for r in results),'rows':results,
             'scope':'Training scans only; gradient correlation across eight orientations. This does not prove subpixel registration.'}
    target = ROOT/'results/hsi_unet_v2' if rotation==3 else ROOT/'results/hsi_unet_v1'
    atomic_json(target/'alignment_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='rows'},indent=2),flush=True)
    if not audit['passed']:
        raise ValueError('PNG/cube alignment needs investigation before training')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',default=str(ROOT/'data/hsi_prepared_v1/manifest.json'))
    parser.add_argument('--rotation',type=int,choices=[0,1,2,3],default=0)
    parser.add_argument('--all-training',action='store_true')
    args=parser.parse_args()
    run(args.manifest,args.rotation,args.all_training)
