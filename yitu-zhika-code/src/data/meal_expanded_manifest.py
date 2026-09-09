"""Versioned dish-level training expansion; historical holdouts stay unchanged."""
import copy
import math
import hashlib


def validate_expansion_labels(expansion, official_train, official_test, metadata):
    for row in expansion['rows']:
        key = row['dish_id']
        if key not in official_train or key in official_test or row['split'] != 'train':
            raise ValueError('Expansion dish is not an official training ID')
        if key not in metadata or any(abs(a-b) > 1e-4 for a, b in zip(row['targets'], metadata[key])):
            raise ValueError('Expansion physical target mismatch')
        day = row['capture_day_utc']
        if int(hashlib.sha256(f'42:{day}'.encode()).hexdigest()[:8], 16)/2**32 < .15:
            raise ValueError('Expansion dish belongs to the frozen validation day partition')


def merge_training_dishes(base, expansion):
    result = copy.deepcopy(base)
    rows = result['rows']
    known = {r['dish_id'] for r in rows}
    if len(known) != len(rows):
        raise ValueError('Duplicate dish IDs in base manifest')
    pixels = {r['pixel_sha256'] for r in rows}
    for source in expansion['rows']:
        if source['split'] != 'train' or source['dish_id'] in known:
            raise ValueError('Expansion must contain new training dishes only')
        frames = source['frames']
        if not frames or len({f['pixel_sha256'] for f in frames}) != len(frames):
            raise ValueError('Missing or duplicated views')
        if any(f['pixel_sha256'] in pixels for f in frames):
            raise ValueError('Expansion overlaps existing image content')
        known.add(source['dish_id'])
        pixels.update(f['pixel_sha256'] for f in frames)
        row = dict(dish_id=source['dish_id'], split='train', targets=source['targets'],
                   category=source['category'], category_idx=result['category_to_idx'][source['category']],
                   image=frames[0]['path'], image_sha256=frames[0]['file_sha256'],
                   pixel_sha256=frames[0]['pixel_sha256'], views=copy.deepcopy(frames),
                   source='official_side_video', capture_day_utc=source.get('capture_day_utc'),
                   cooking_method=None, retained_oil_grams=None, mass_basis='as_served_dish')
        rows.append(row)
    for row in rows:
        if len(row['targets']) != 2 or not all(math.isfinite(v) and v >= 0 for v in row['targets']):
            raise ValueError('Invalid physical targets')
    training = [r['targets'] for r in rows if r['split'] == 'train']
    if not training:
        raise ValueError('Empty training split')
    means = [sum(t[i] for t in training)/len(training) for i in range(2)]
    scales = [(sum((t[i]-means[i])**2 for t in training)/len(training))**.5 for i in range(2)]
    if min(scales) <= 0:
        raise ValueError('Degenerate training target variance')
    result.update(protocol='nutrition5k_expanded_dishes_v2',
                  counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')},
                  target_stats=dict(mean=means, std=scales, source='train_only', samples=len(training)),
                  expansion_dishes=len(expansion['rows']))
    return result
