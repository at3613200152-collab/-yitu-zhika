"""Verify frozen inputs and publish a new dish-level manifest; no downloads."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest
from src.data.meal_expanded_manifest import merge_training_dishes, validate_expansion_labels

BASE = ROOT/'results/meal_official_v1/manifest.json'
BASE_SHA = 'f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa'
EXTRA_SHA = '384da310c8bd644b137454ccf261b93e8a0df690448d24aa22c0d25c207e128b'
OUTPUT = ROOT/'results/meal_expanded_v2'


def run():
    pointer = json.loads((ROOT/'results/video_expansion_v1.json').read_text(encoding='utf-8'))
    extra_path = Path(pointer['manifest'])
    if file_digest(BASE) != BASE_SHA or file_digest(extra_path) != EXTRA_SHA:
        raise ValueError('Frozen source manifest hash mismatch')
    base = json.loads(BASE.read_text(encoding='utf-8'))
    extra = json.loads(extra_path.read_text(encoding='utf-8'))
    if not extra['complete'] or len(extra['rows']) != 128:
        raise ValueError('Expected all 128 audited expansion dishes')
    meta_path = ROOT/'data/Nutrition5k/dishes_verified.csv'
    if file_digest(meta_path) != base['metadata_sha256']:
        raise ValueError('Verified nutritional metadata changed')
    with meta_path.open(encoding='utf-8-sig', newline='') as stream:
        metadata = {r['dish_id']: [float(r['total_calories']), float(r['total_mass'])]
                    for r in csv.DictReader(stream)}
    ids = {}
    for split in ('train', 'test'):
        path = ROOT/f'data/Nutrition5k/official_metadata/rgb_{split}_ids.txt'
        if file_digest(path) != base['official_sources'][split]['sha256']:
            raise ValueError('Official IDs changed')
        ids[split] = set(path.read_text().split())
    validate_expansion_labels(extra, ids['train'], ids['test'], metadata)
    manifest = merge_training_dishes(base, extra)
    manifest['sources'] = dict(overhead_manifest_sha256=BASE_SHA, expansion_manifest_sha256=EXTRA_SHA,
                               metadata_sha256=base['metadata_sha256'])
    manifest['limitations'] = base['limitations'] + [
        'Same-dish views are correlated; 128 extra dishes, not 384 independent meals.',
        'Frozen v1 weak coarse-category labels retained across all arms for comparability; known keyword errors remain.',
        'Cooking method and retained oil are unknown; density is derived kcal/as-served grams, not a cooking label.',
        'Fixed evaluation is overhead-only; expanded training views are not an independent side-view test.']
    # Density supervision is derived only from available physical targets.
    ratios = [r['targets'][0]/r['targets'][1] for r in manifest['rows']
              if r['split'] == 'train' and r['targets'][1] > 0]
    manifest['density_stats'] = dict(mean=sum(ratios)/len(ratios), samples=len(ratios), source='train_only')
    images = 0
    for row in manifest['rows']:
        expected = metadata[row['dish_id']]
        if any(abs(a-b) > 1e-4 for a, b in zip(row['targets'], expected)):
            raise ValueError('Base/expansion metadata disagreement')
        views = row.get('views', [dict(path=row['image'], file_sha256=row['image_sha256'])])
        for view in views:
            path = ROOT/view['path']
            if file_digest(path) != view['file_sha256']:
                raise ValueError('Image hash changed: '+row['dish_id'])
            images += 1
        if images % 500 == 0:
            print(json.dumps(dict(stage='verifying_images', images=images)), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT/'manifest.json'
    if dest.exists() and json.loads(dest.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Expanded manifest already exists with different content; use a new version')
    atomic_json(dest, manifest)
    audit = dict(passed=True, manifest_sha256=file_digest(dest), counts=manifest['counts'],
                 additional_dishes=128, additional_views=384, verified_images=images,
                 holdout_rows_unchanged=True, cooking_labels_added=0,
                 actual_training_started=False, sources=manifest['sources'])
    atomic_json(OUTPUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    run()
