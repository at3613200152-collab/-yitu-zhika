"""Audit converted Nutrition5k labels against the official raw CSVs.

Creates a separate verified CSV; never rewrites the original converted data.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FIELDS = ('total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein')
ROOT = Path(__file__).resolve().parents[2]
LEGACY_CHECKPOINT_SHA256 = '38020a6463cea8c3b4305d3c77b8c47d35ea88f19d6abbf4564027cdc63a4741'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read_official(paths):
    rows = {}
    for path in paths:
        with Path(path).open(encoding='utf-8', newline='') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if not row[0].startswith('dish_') or len(row) < 6:
                    raise ValueError('Unexpected official metadata row')
                if row[0] in rows:
                    raise ValueError(f'Duplicate official ID: {row[0]}')
                values = dict(zip(FIELDS, map(float, row[1:6])))
                if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                    raise ValueError(f'Invalid official values: {row[0]}')
                rows[row[0]] = values
    return rows


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-5)


def legacy_checkpoint_metadata(checkpoint):
    # Never attach historical semantics to an arbitrary replacement at the same path.
    if digest(checkpoint) != LEGACY_CHECKPOINT_SHA256:
        raise ValueError('Not the audited historical checkpoint; explicit new-model metadata required')
    return {'schema': 1, 'checkpoint_sha256': LEGACY_CHECKPOINT_SHA256,
            'target_names': ['mass', 'calories'], 'classification_valid': False,
            'category_to_idx': {}, 'evidence': 'results/metadata_audit/nutrition5k_audit.json',
            'limitations': 'Historical target semantics reconstructed from training code and converted CSV. Original checkpoint lacks a training-data hash; historical split is not the official benchmark split.'}


def audit(local, official):
    if len({r['dish_id'] for r in local}) != len(local):
        raise ValueError('Duplicate local IDs')
    if set(r['dish_id'] for r in local) != set(official):
        raise ValueError('Official and local ID sets differ')
    swapped, as_named, macro_bad = 0, 0, 0
    for row in local:
        truth = official[row['dish_id']]
        swapped += close(row['total_mass'], truth['total_calories']) and close(row['total_calories'], truth['total_mass'])
        as_named += close(row['total_mass'], truth['total_mass']) and close(row['total_calories'], truth['total_calories'])
        macro_bad += any(not close(row[k], truth[k]) for k in FIELDS[2:])
    return {'rows': len(local), 'swapped_matches': swapped, 'as_named_matches': as_named,
            'macro_mismatches': macro_bad,
            'official_nonpositive_calories': sum(r['total_calories'] <= 0 for r in official.values())}


def main(write=False):
    data = ROOT/'data/Nutrition5k'
    source_paths = [data/'official_metadata'/f'dish_metadata_cafe{i}.csv' for i in (1,2)]
    official = read_official(source_paths)
    with (data/'dishes.csv').open(encoding='utf-8', newline='') as f:
        local = list(csv.DictReader(f))
    result = audit(local, official)
    result['source_urls'] = [f'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/{p.name}' for p in source_paths]
    result['sha256'] = {p.name: digest(p) for p in [*source_paths, data/'dishes.csv', data/'dishes_categorized.csv']}
    result['image_count'] = len(list((data/'images').glob('*_rgb.jpg')))
    for split in ('train', 'test'):
        p = data/'official_metadata'/f'depth_{split}_ids.txt'
        if p.exists():
            ids = p.read_text(encoding='utf-8').splitlines()
            result[f'official_depth_{split}_ids'] = len(ids)
            result[f'official_depth_{split}_images_available'] = sum((data/'images'/f'{i}_rgb.jpg').exists() for i in ids)
    if write:
        if result['swapped_matches'] != len(local) or result['macro_mismatches']:
            raise ValueError('Whole-file swap not proven; refusing automatic migration')
        with (data/'dishes_categorized.csv').open(encoding='utf-8', newline='') as f:
            categories = {r['dish_id']:r['category'] for r in csv.DictReader(f)}
        target = data/'dishes_verified.csv'
        columns = ['dish_id', 'total_mass', 'total_calories', 'total_fat', 'total_carb', 'total_protein', 'category']
        with target.with_suffix('.csv.tmp').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            # Keep original ordering so historical random splits remain reproducible.
            for row in local:
                dish_id = row['dish_id']
                writer.writerow({'dish_id':dish_id, **official[dish_id], 'category':categories[dish_id]})
        target.with_suffix('.csv.tmp').replace(target)
        result['verified_csv_sha256'] = digest(target)
        report = ROOT/'results/metadata_audit'
        report.mkdir(parents=True, exist_ok=True)
        (report/'nutrition5k_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        # This mapping applies only to this exact historical checkpoint, not every 2-head model.
        checkpoint = ROOT/'checkpoints/multitask/best_model.pt'
        if checkpoint.exists() and digest(checkpoint) == LEGACY_CHECKPOINT_SHA256:
            sidecar = legacy_checkpoint_metadata(checkpoint)
            Path(str(checkpoint)+'.metadata.json').write_text(json.dumps(sidecar, indent=2), encoding='utf-8')
        else:
            print('Verified CSV saved; unknown/missing checkpoint metadata left untouched.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)
