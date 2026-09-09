"""Frozen official Nutrition5k RGB-ID split, local overhead subset protocol."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, file_digest

PROTOCOL = 'nutrition5k_overhead_official_ids_v1'
TARGETS = ['calories', 'mass']


def capture_day(dish_id):
    return datetime.fromtimestamp(int(dish_id.removeprefix('dish_')), timezone.utc).date().isoformat()


def training_partition(dish_id, seed=42):
    # Conservative capture-day grouping; no claim of ground-truth plate grouping.
    key = f'{seed}:{capture_day(dish_id)}'.encode()
    score = int(hashlib.sha256(key).hexdigest()[:8], 16) / 2**32
    return 'val' if score < 0.15 else 'train'


def assign_split(dish_id, official_train, official_test):
    if official_train & official_test:
        raise ValueError('Official train/test IDs overlap')
    if dish_id in official_test:
        return 'test'
    if dish_id in official_train:
        return training_partition(dish_id)
    return 'excluded'


def target_stats(rows):
    values = [r['targets'] for r in rows if r['split'] == 'train']
    if len(values) < 2:
        raise ValueError('Insufficient training samples')
    mean = [sum(v[i] for v in values)/len(values) for i in range(len(TARGETS))]
    std = [math.sqrt(sum((v[i]-mean[i])**2 for v in values)/len(values)) for i in range(len(TARGETS))]
    if not all(math.isfinite(s) and s > 0 for s in std):
        raise ValueError('Degenerate training target statistics')
    return {'mean':mean, 'std':std, 'source':'train_only', 'samples':len(values)}


def fetch_split(data, split):
    name = f'rgb_{split}_ids.txt'
    path = data/'official_metadata'/name
    url = f'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/{name}'
    response = requests.get(url, timeout=(10, 30))
    response.raise_for_status()
    content = response.content
    if len(content) > 200000 or not content.startswith(b'dish_'):
        raise ValueError('Unexpected split response')
    ids = content.decode('utf-8').splitlines()
    if len(ids) != len(set(ids)) or not all(i.startswith('dish_') and i[5:].isdigit() for i in ids):
        raise ValueError('Invalid official IDs')
    if path.exists() and path.read_bytes() != content:
        raise ValueError('Official split changed; refusing implicit replacement')
    if not path.exists():
        path.write_bytes(content)
    return set(ids), {'url':url, 'sha256':file_digest(path), 'count':len(ids)}


def build():
    data = ROOT/'data/Nutrition5k'
    output = ROOT/'results/meal_official_v1/manifest.json'
    check_space(ROOT)
    audit = json.loads((ROOT/'results/metadata_audit/nutrition5k_audit.json').read_text(encoding='utf-8'))
    csv_path = data/'dishes_verified.csv'
    if file_digest(csv_path) != audit['verified_csv_sha256']:
        raise ValueError('Verified metadata differs from audited hash')
    with csv_path.open(encoding='utf-8', newline='') as f:
        metadata = list(csv.DictReader(f))
    official_train, train_source = fetch_split(data, 'train')
    official_test, test_source = fetch_split(data, 'test')
    categories = {c:i for i,c in enumerate(sorted({r['category'] for r in metadata}))}
    rows, excluded, seen_pixels = [], [], {}
    for index, row in enumerate(metadata):
        dish_id = row['dish_id']
        split = assign_split(dish_id, official_train, official_test)
        image_path = data/'images'/f'{dish_id}_rgb.jpg'
        if not image_path.exists() or split == 'excluded':
            excluded.append({'dish_id':dish_id, 'reason':'no_local_overhead' if not image_path.exists() else 'not_in_official_rgb_ids'})
            continue
        targets = [float(row['total_calories']), float(row['total_mass'])]
        if not all(math.isfinite(v) and v >= 0 for v in targets) or targets[1] <= 0:
            raise ValueError(f'Invalid targets: {dish_id}')
        with Image.open(image_path) as im:
            im = im.convert('RGB')
            im.load()
            geometry = list(im.size)
            pixels = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        if pixels in seen_pixels and seen_pixels[pixels]['split'] != split:
            raise ValueError(f'Exact image leakage: {seen_pixels[pixels]} versus {dish_id}/{split}')
        seen_pixels[pixels] = {'dish_id':dish_id, 'split':split}
        rows.append({'dish_id':dish_id, 'split':split, 'capture_day_utc':capture_day(dish_id),
                     'image':str(image_path.relative_to(ROOT)), 'image_sha256':file_digest(image_path),
                     'pixel_sha256':pixels, 'size':geometry, 'targets':targets,
                     'category':row['category'], 'category_idx':categories[row['category']]})
        if index % 300 == 0:
            print(json.dumps({'metadata_rows_checked':index+1, 'usable_images':len(rows)}), flush=True)
    counts = dict(Counter(r['split'] for r in rows))
    if any(counts.get(s,0) < 100 for s in ('train','val','test')):
        raise ValueError(f'Unexpected split counts: {counts}')
    manifest = {'protocol':PROTOCOL, 'metadata_sha256':file_digest(csv_path),
                'official_sources':{'train':train_source, 'test':test_source},
                'target_names':TARGETS, 'target_units':['kcal','g'],
                'category_to_idx':categories, 'category_source':'derived_ingredient_mass_keyword_groups_not_official_food_classes',
                'target_stats':target_stats(rows), 'counts':counts,
                'category_counts':{s:dict(Counter(r['category'] for r in rows if r['split']==s)) for s in counts},
                'zero_calories':{s:sum(r['targets'][0]==0 for r in rows if r['split']==s) for s in counts},
                'validation_rule':'SHA256(42:UTC_capture_day)<0.15; drawn only from official training IDs',
                'limitations':['Local overhead-image subset, not full official multi-view benchmark.',
                    'Capture-day validation grouping is a proxy; exact plate IDs and near-duplicate independence are not fully known.',
                    'Zero-calorie labels retained; excluded only from percentage division, never from MAE/RMSE.',
                    'Historical random-split weights cannot be evaluated as clean official-test baselines.'],
                'rows':rows, 'excluded':excluded}
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError('Frozen manifest differs; a new protocol version is required')
    else:
        atomic_json(output, manifest)
    print(json.dumps({'manifest':str(output), 'sha256':file_digest(output), 'counts':counts,
                      'categories':manifest['category_counts'], 'zero_calories':manifest['zero_calories'],
                      'target_stats':manifest['target_stats']}, indent=2), flush=True)


if __name__ == '__main__':
    build()
