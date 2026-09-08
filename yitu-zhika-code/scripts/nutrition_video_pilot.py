"""Bounded official side-video download/decoding pilot; never edits training splits."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import cv2
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import (atomic_json, check_space, check_stop, file_digest,
                                 job_lock, resume_download, verify_archive)
from src.data.meal_manifest import training_partition

API = 'https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o'
DATA = ROOT/'data/Nutrition5k'
MAX_BYTES = 256 * 1024**2


def select_objects(count):
    audited = json.loads((ROOT/'results/metadata_audit/nutrition5k_audit.json').read_text(encoding='utf-8'))
    if file_digest(DATA/'dishes_verified.csv') != audited['verified_csv_sha256']:
        raise ValueError('Nutrition metadata has changed')
    with (DATA/'dishes_verified.csv').open(encoding='utf-8', newline='') as f:
        rows = {r['dish_id']:r for r in csv.DictReader(f)}
    ids = (DATA/'official_metadata/rgb_train_ids.txt').read_text().splitlines()
    candidates = sorted((d for d in ids if d in rows and training_partition(d)=='train'
                         and not (DATA/'images'/f'{d}_rgb.jpg').exists()),
                        key=lambda d: hashlib.sha256(d.encode()).hexdigest())
    selected = []
    for dish in candidates[:count+20]:
        prefix = f'nutrition5k_dataset/imagery/side_angles/{dish}/'
        response = requests.get(API, params={'prefix':prefix, 'maxResults':100,
            'fields':'items(name,size,md5Hash,generation),nextPageToken'}, timeout=(15,30))
        response.raise_for_status()
        listing = response.json()
        if listing.get('nextPageToken'):
            raise ValueError('Unexpected per-dish pagination')
        files = sorted((o for o in listing.get('items',[]) if o['name'].endswith('.h264')),
                       key=lambda o:o['name'])
        if not files:
            continue
        obj = files[0]  # Deterministic camera order, not chosen by test performance.
        if int(obj['size']) > 64*1024**2:
            raise ValueError('Pilot per-video size cap exceeded')
        label = rows[dish]
        selected.append({'dish_id':dish, 'split':'train', 'object':obj,
                         'targets':[float(label['total_calories']),float(label['total_mass'])],
                         'category':label['category']})
        if len(selected) == count:
            break
    if len(selected) != count or sum(int(r['object']['size']) for r in selected)>MAX_BYTES:
        raise ValueError('Pilot selection/count/size gate failed')
    return {'protocol':'side_angle_pilot_v1', 'metadata_sha256':audited['verified_csv_sha256'],
            'source':'https://github.com/google-research-datasets/Nutrition5k',
            'purpose':'Decode and label-link validation only; not merged into frozen overhead experiment',
            'target_names':['calories','mass'], 'target_units':['kcal','g'],
            'frame_indices':[0,30,60], 'rows':selected}


def extract_frames(video, folder, indices):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('OpenCV cannot decode raw H264')
    frames = []
    try:
        for i in range(max(indices)+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Video ended before frame {i}')
            if i not in indices:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if min(rgb.shape[:2]) < 128 or float(rgb.std()) < 2:
                raise ValueError('Invalid/constant video frame')
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            output = folder/f'frame_{i:04d}.png'
            # Store full decoded resolution; derived frames are reproducible.
            Image.fromarray(rgb).save(output)
            frames.append({'index':i, 'path':str(output), 'size':[rgb.shape[1],rgb.shape[0]],
                           'pixel_sha256':digest, 'file_sha256':file_digest(output)})
    finally:
        cap.release()
    if len({f['pixel_sha256'] for f in frames}) != len(indices):
        raise ValueError('Duplicate frames in pilot')
    return frames


def run(destination, count):
    destination = Path(destination).resolve()
    if not 1 <= count <= 4:
        raise ValueError('Pilot allows one to four additional dishes only')
    if destination != Path('D:/yitu-data/Nutrition5k/side_angle_pilot_v1').resolve():
        raise ValueError('This pilot writes only to the explicitly scoped D-drive directory')
    destination.mkdir(parents=True, exist_ok=True)
    check_space(destination)
    state = {'pid':os.getpid(), 'destination':str(destination), 'dishes_requested':count}
    def report(**kwargs):
        check_stop(destination)
        state.update(kwargs, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(destination/'status.json',state)
        print(json.dumps(kwargs,ensure_ascii=False),flush=True)
    with job_lock(destination/'job.lock'):
        try:
            report(stage='selecting')
            plan_path = destination/'download_plan.json'
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding='utf-8'))
                if len(plan['rows']) != count:
                    raise ValueError('Frozen pilot count differs')
            else:
                plan = select_objects(count)
                atomic_json(plan_path,plan)
            completed = []
            for row in plan['rows']:
                obj = row['object']
                folder = destination/row['dish_id']
                folder.mkdir(exist_ok=True)
                video = folder/Path(obj['name']).name
                # Generation pin prevents mixing file revisions during resume.
                url = 'https://storage.googleapis.com/nutrition5k_dataset/'+quote(obj['name'],safe='/')
                url += '?generation='+obj['generation']
                report(dish_id=row['dish_id'], dishes_done=len(completed))
                resume_download(url, video, int(obj['size']), report,
                                max_seconds=1200, max_no_progress=5)
                report(stage='checksum')
                md5 = base64.b64decode(obj['md5Hash'],validate=True).hex()
                verify_archive(video,int(obj['size']),md5)
                report(stage='decoding')
                frames = extract_frames(video,folder,plan['frame_indices'])
                completed.append({**row,'video':str(video),'verified_md5':md5,'frames':frames})
                atomic_json(destination/'manifest.json',{**plan,'rows':completed,'complete':False})
            manifest = {**plan,'rows':completed,'complete':True,
                        'independent_dish_ids':len(completed),'decoded_frames':sum(len(r['frames']) for r in completed)}
            atomic_json(destination/'manifest.json',manifest)
            atomic_json(ROOT/'results/video_expansion_pilot.json',{
                'manifest':str(destination/'manifest.json'),'sha256':file_digest(destination/'manifest.json'),
                'dishes':len(completed),'frames':manifest['decoded_frames'],'included_in_training':False})
            report(stage='complete',dishes_done=len(completed),frames=manifest['decoded_frames'])
        except BaseException as exc:
            state.update(stage='paused' if isinstance(exc,InterruptedError) else 'failed',
                         error_type=type(exc).__name__, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(destination/'status.json',state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination',default='D:/yitu-data/Nutrition5k/side_angle_pilot_v1')
    parser.add_argument('--count',type=int,default=4)
    args = parser.parse_args()
    run(args.destination,args.count)
