"""Bounded, resumable Nutrition5k expansion, separate from frozen overhead training."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest, job_lock
from scripts.nutrition_video_pilot import extract_frames
from src.data.meal_manifest import capture_day, training_partition

DATA = ROOT / 'data/Nutrition5k'
DESTINATION = Path('D:/yitu-data/Nutrition5k/side_angle_expansion_v1')
PILOT = Path('D:/yitu-data/Nutrition5k/side_angle_pilot_v1')
API = 'https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o'
PROTOCOL = 'side_angle_expansion_v1'
COUNT, MAX_TOTAL, MAX_VIDEO = 128, 4 * 1024**3, 64 * 1024**2
FRAME_INDICES = [0, 30, 60]
FRAME_POLICY = 'first_61_or_short_video_three_evenly_spaced_v2'


def select_frame_indices(count):
    if count < 3:
        raise ValueError('Video has fewer than three decodable frames')
    last = min(count,61)-1
    return [0,last//2,last]


def extract_expansion_frames(video,folder):
    import cv2
    import numpy as np
    from PIL import Image
    cap=cv2.VideoCapture(str(video))
    frames=[]
    try:
        for index in range(61):
            ok,frame=cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    selected=select_frame_indices(len(frames))
    result=[]
    for index in selected:
        rgb=cv2.cvtColor(frames[index],cv2.COLOR_BGR2RGB)
        if min(rgb.shape[:2])<128 or not np.isfinite(rgb).all() or float(rgb.std())<2:
            raise ValueError('Invalid decoded video frame')
        output=folder/f'frame_{index:04d}.png'
        Image.fromarray(rgb).save(output)
        result.append({'index':index,'path':str(output),'size':[rgb.shape[1],rgb.shape[0]],
                       'pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                       'file_sha256':file_digest(output),'decoded_count_up_to_61':len(frames)})
    if len({r['pixel_sha256'] for r in result})!=3:
        raise ValueError('Duplicate decoded frames')
    return result


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def scoped(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes dataset directory')
    return path


def metadata_context():
    audit = json.loads((ROOT / 'results/metadata_audit/nutrition5k_audit.json').read_text(encoding='utf-8'))
    overhead = json.loads((ROOT / 'results/meal_official_v1/manifest.json').read_text(encoding='utf-8'))
    csv_hash = file_digest(DATA / 'dishes_verified.csv')
    if csv_hash != audit['verified_csv_sha256'] or csv_hash != overhead['metadata_sha256']:
        raise ValueError('Audited nutrition metadata changed')
    ids = {}
    for split in ('train', 'test'):
        path = DATA / 'official_metadata' / f'rgb_{split}_ids.txt'
        if file_digest(path) != overhead['official_sources'][split]['sha256']:
            raise ValueError('Official split differs from frozen overhead experiment')
        lines = path.read_text(encoding='utf-8').splitlines()
        if len(lines) != len(set(lines)) or not all(re.fullmatch(r'dish_\d+', d) for d in lines):
            raise ValueError('Invalid official dish IDs')
        ids[split] = set(lines)
    if ids['train'] & ids['test']:
        raise ValueError('Official train/test overlap')
    with (DATA / 'dishes_verified.csv').open(encoding='utf-8', newline='') as stream:
        items = list(csv.DictReader(stream))
    labels = {row['dish_id']: row for row in items}
    if len(labels) != len(items):
        raise ValueError('Duplicate metadata dish IDs')
    candidates = sorted((d for d in ids['train'] if d in labels and training_partition(d) == 'train'
                         and not (DATA / 'images' / f'{d}_rgb.jpg').exists()),
                        key=lambda d: hashlib.sha256(d.encode()).hexdigest())
    sources = {'metadata_sha256': csv_hash,
               'overhead_manifest_sha256': file_digest(ROOT / 'results/meal_official_v1/manifest.json'),
               'official_sources': overhead['official_sources'], 'candidate_ids_sha256': digest_json(candidates)}
    return labels, candidates, sources


def choose_object(dish, listing):
    prefix = f'nutrition5k_dataset/imagery/side_angles/{dish}/'
    if listing.get('nextPageToken'):
        raise ValueError('Unexpected per-dish pagination')
    files = [o for o in listing.get('items', []) if o['name'].endswith('.h264')]
    for obj in files:
        name = obj['name']
        if not name.startswith(prefix) or '/' in name[len(prefix):] or '\\' in name or '..' in name:
            raise ValueError('Unexpected cloud object path')
    files.sort(key=lambda o: (PurePosixPath(o['name']).name != 'camera_A.h264', o['name']))
    if not files:
        return None
    obj = {k: str(files[0][k]) for k in ('name', 'generation', 'size', 'md5Hash')}
    if not obj['generation'].isdigit() or int(obj['generation']) <= 0 or int(obj['size']) <= 0:
        raise ValueError('Invalid generation or object size')
    if len(base64.b64decode(obj['md5Hash'], validate=True)) != 16:
        raise ValueError('Invalid official MD5')
    return obj


def wait_checked(seconds, guard):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        guard()
        time.sleep(min(1, max(0, until - time.monotonic())))


def fetch_listing(dish, guard):
    for attempt in range(1, 5):
        guard()
        try:
            with requests.get(API, params={'prefix': f'nutrition5k_dataset/imagery/side_angles/{dish}/',
                    'maxResults': 100, 'fields': 'items(name,size,md5Hash,generation),nextPageToken'},
                    timeout=(10, 20)) as response:
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    raise requests.ConnectionError('Transient listing response')
                response.raise_for_status()
                return response.json()
        except requests.RequestException as exc:
            if isinstance(exc, requests.HTTPError) or attempt == 4:
                raise RuntimeError(f'Cloud listing failed: {type(exc).__name__}') from None
            wait_checked(5 * attempt, guard)


def build_plan(destination, labels, candidates, sources, guard, report):
    path = destination / 'selection_progress.json'
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'sources': sources, 'objects': {}}
    if cache['sources'] != sources:
        raise ValueError('Selection inputs changed; use a new expansion version')
    rows, excluded = [], []
    for dish in candidates:
        guard()
        if dish not in cache['objects']:
            cache['objects'][dish] = choose_object(dish, fetch_listing(dish, guard))
            atomic_json(path, cache)
        obj = cache['objects'][dish]
        if obj is None or int(obj['size']) > MAX_VIDEO:
            excluded.append({'dish_id': dish, 'reason': 'no_video' if obj is None else 'preferred_video_exceeds_64MiB'})
            continue
        label = labels[dish]
        targets = [float(label['total_calories']), float(label['total_mass'])]
        if not all(math.isfinite(v) and v >= 0 for v in targets) or targets[1] <= 0:
            raise ValueError(f'Invalid physical targets: {dish}')
        rows.append({'dish_id': dish, 'split': 'train', 'capture_day_utc': capture_day(dish),
                     'object': obj, 'targets': targets, 'category': label['category']})
        report(stage='selecting', selected=len(rows), candidates_examined=len(rows) + len(excluded))
        if len(rows) == COUNT:
            break
    total = sum(int(row['object']['size']) for row in rows)
    if len(rows) != COUNT or total > MAX_TOTAL:
        raise ValueError(f'Selection fails count/4GiB gate: {len(rows)} dishes, {total} bytes')
    return {'protocol': PROTOCOL, 'sources': sources, 'rows': rows, 'excluded_candidates': excluded,
            'planned_video_bytes': total, 'max_total_video_bytes': MAX_TOTAL, 'max_video_bytes': MAX_VIDEO,
            'frame_indices': FRAME_INDICES, 'target_names': ['calories', 'mass'], 'target_units': ['kcal', 'g'],
            'selection_rule': 'SHA256(dish_id) order; official train plus UTC-day train; missing overhead; camera_A then filename; skip absent or >64MiB preferred videos',
            'source': 'https://github.com/google-research-datasets/Nutrition5k', 'included_in_training': False,
            'limitations': ['Frames/views of one dish are correlated, not independent meals.',
                'Capture-day grouping is a proxy; physical-plate independence is not established.',
                'Side-view expansion requires a separate future training/view protocol.']}


def validate_plan(plan, candidates, sources, labels):
    if plan['protocol'] != PROTOCOL or plan['sources'] != sources or plan['frame_indices'] != FRAME_INDICES:
        raise ValueError('Frozen expansion protocol changed')
    rows = plan['rows']
    ids = [row['dish_id'] for row in rows]
    if len(ids) != COUNT or len(set(ids)) != COUNT or not set(ids).issubset(candidates):
        raise ValueError('Expansion requires 128 unique eligible training dishes')
    sizes = [int(row['object']['size']) for row in rows]
    if any(s <= 0 or s > MAX_VIDEO for s in sizes) or sum(sizes) > MAX_TOTAL or sum(sizes) != plan['planned_video_bytes']:
        raise ValueError('Video size budget exceeded or changed')
    for row in rows:
        dish = row['dish_id']
        label = labels[dish]
        if row['split'] != 'train' or training_partition(dish) != 'train' or row['capture_day_utc'] != capture_day(dish):
            raise ValueError('Expansion crosses frozen train partition')
        if row['targets'] != [float(label['total_calories']), float(label['total_mass'])] or row['category'] != label['category']:
            raise ValueError('Frozen labels differ from audited metadata')
        if choose_object(dish, {'items': [row['object']]}) != row['object']:
            raise ValueError('Invalid frozen object')


def download_video(obj, video, guard, report):
    """Generation-pinned ranges; eight attempts and 20 minutes per video."""
    expected = int(obj['size'])
    url = 'https://storage.googleapis.com/nutrition5k_dataset/' + quote(obj['name'], safe='/')
    url += '?generation=' + obj['generation']
    deadline, last_report = time.monotonic() + 1200, 0
    for attempt in range(1, 9):
        guard()
        offset = video.stat().st_size if video.exists() else 0
        if offset > expected:
            raise ValueError('Existing video exceeds official size; retained')
        if offset == expected:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Per-video deadline exceeded; partial retained')
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'YituZhika-expansion/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        report(stage='downloading', bytes=offset, total_bytes=expected, attempt=attempt)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 20)) as response:
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    raise requests.ConnectionError('Transient video response')
                if response.status_code not in (200, 206):
                    raise ValueError(f'Video HTTP {response.status_code}; source requires review')
                if offset or response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status_code != 206 or not match:
                        raise ValueError('Server ignored Range; refusing append')
                    first, last, total = map(int, match.groups())
                    if first != offset or total != expected or not first <= last < total:
                        raise ValueError('Unexpected Content-Range')
                elif int(response.headers.get('Content-Length', '-1')) != expected:
                    raise ValueError('Unexpected content length')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise ValueError('Encoded HTTP body invalidates offsets')
                with video.open('ab') as stream:
                    for chunk in response.iter_content(chunk_size=65536):
                        guard()
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Per-video deadline exceeded; partial retained')
                        if not chunk:
                            continue
                        if stream.tell() + len(chunk) > expected:
                            raise ValueError('Response exceeds official size')
                        stream.write(chunk)
                        if time.monotonic() - last_report >= 10:
                            stream.flush()
                            report(stage='downloading', bytes=stream.tell(), total_bytes=expected, attempt=attempt)
                            last_report = time.monotonic()
            if video.stat().st_size == expected:
                return
        except requests.RequestException as exc:
            report(stage='retrying', attempt=attempt, network_error_type=type(exc).__name__)
        if attempt < 8:
            wait_checked(min(30, 5 * attempt), guard)
    raise RuntimeError('Eight video attempts exhausted; partial retained')


def validate_receipt(receipt, row, allowed_root, guard):
    if any(receipt.get(k) != row[k] for k in row):
        raise ValueError('Completed metadata differs from frozen plan')
    video = scoped(receipt['video'], allowed_root)
    guard()
    md5 = base64.b64decode(row['object']['md5Hash'], validate=True).hex()
    if video.stat().st_size != int(row['object']['size']) or file_digest(video, 'md5') != md5:
        raise ValueError('Completed video checksum differs; retained for inspection')
    expected_indices = (select_frame_indices(receipt['frames'][0]['decoded_count_up_to_61'])
                        if receipt.get('decode_policy') == FRAME_POLICY else FRAME_INDICES)
    if receipt['verified_md5'] != md5 or [f['index'] for f in receipt['frames']] != expected_indices:
        raise ValueError('Completed frame sequence differs')
    for frame in receipt['frames']:
        guard()
        if file_digest(scoped(frame['path'], allowed_root)) != frame['file_sha256']:
            raise ValueError('Completed frame file checksum differs')
    if len({f['pixel_sha256'] for f in receipt['frames']}) != 3:
        raise ValueError('Completed frames are duplicates')
    return receipt


def load_pilot_rows():
    pointer = json.loads((ROOT / 'results/video_expansion_pilot.json').read_text(encoding='utf-8'))
    path = scoped(pointer['manifest'], PILOT)
    if file_digest(path) != pointer['sha256']:
        raise ValueError('Pilot manifest hash differs from verified project pointer')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not manifest.get('complete') or manifest['frame_indices'] != FRAME_INDICES:
        raise ValueError('Pilot is incomplete or incompatible')
    return {row['dish_id']: row for row in manifest['rows']}


def decode_bounded(video, folder, guard):
    command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()),
               '--decode-video', str(video), '--decode-folder', str(folder)]
    with (folder / 'decode.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + 120
        try:
            while child.poll() is None:
                guard()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Video decode exceeded 120 seconds')
                time.sleep(0.5)
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise
        if child.returncode:
            raise ValueError('Video decoding failed; inspect per-dish decode.log')
    return json.loads((folder / 'decoded_frames.json').read_text(encoding='utf-8'))


def run(destination, plan_only=False, retry_failed=False):
    destination = Path(destination).resolve()
    if destination != DESTINATION.resolve():
        raise ValueError('Expansion writes only to its scoped D-drive directory')
    destination.mkdir(parents=True, exist_ok=True)
    with job_lock(destination / 'job.lock'):
        state_path = destination / 'status.json'
        previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        if previous.get('stage') == 'failed' and not retry_failed:
            raise RuntimeError('Inspect failure.json before explicit --retry-failed')
        budget_path = destination / 'run_budget.json'
        budget = json.loads(budget_path.read_text(encoding='utf-8')) if budget_path.exists() else None
        # Planning does not start the download budget; resumption never resets it.
        if budget is None and not plan_only:
            budget = {'started_at': now(), 'deadline_unix': time.time() + 8 * 3600, 'total_hours': 8}
            atomic_json(budget_path, budget)
        budget = budget or {'started_at': now(), 'deadline_unix': time.time() + 1800, 'total_hours': 0.5}
        state = {'protocol': PROTOCOL, 'pid': os.getpid(), 'destination': str(destination), **budget}
        last_space = 0
        def guard():
            nonlocal last_space
            check_stop(destination)
            if time.time() >= budget['deadline_unix']:
                raise TimeoutError('Expansion deadline reached; files retained')
            if time.monotonic() - last_space >= 5:
                check_space(destination)
                check_space(ROOT)
                last_space = time.monotonic()
        def report(**values):
            guard()
            state.update(values, updated_at=now())
            atomic_json(state_path, state)
            print(json.dumps(values, ensure_ascii=False), flush=True)
        try:
            guard()
            labels, candidates, sources = metadata_context()
            plan_path = destination / 'download_plan.json'
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding='utf-8'))
            else:
                report(stage='selecting')
                plan = build_plan(destination, labels, candidates, sources, guard, report)
                atomic_json(plan_path, plan)
            validate_plan(plan, candidates, sources, labels)
            plan_sha = file_digest(plan_path)
            if plan_only:
                report(stage='plan_ready', selected=COUNT, video_bytes=plan['planned_video_bytes'], plan_sha256=plan_sha)
                return
            pilot_rows, completed = load_pilot_rows(), []
            for row in plan['rows']:
                dish = row['dish_id']
                report(stage='checking', dish_id=dish, dishes_done=len(completed), dishes_requested=COUNT)
                folder = destination / dish
                folder.mkdir(exist_ok=True)
                receipt_path = folder / 'completed.json'
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                    allowed_root = PILOT if receipt.get('storage') == 'verified_pilot_reference' else destination
                    validate_receipt(receipt, row, allowed_root, guard)
                elif dish in pilot_rows and pilot_rows[dish]['object'] == row['object']:
                    receipt = {**pilot_rows[dish], **row, 'storage': 'verified_pilot_reference'}
                    validate_receipt(receipt, row, PILOT, guard)
                    atomic_json(receipt_path, receipt)
                else:
                    video = folder / PurePosixPath(row['object']['name']).name
                    download_video(row['object'], video, guard, report)
                    report(stage='checksum')
                    md5 = base64.b64decode(row['object']['md5Hash'], validate=True).hex()
                    if video.stat().st_size != int(row['object']['size']) or file_digest(video, 'md5') != md5:
                        raise ValueError('Downloaded video checksum mismatch; bytes retained')
                    report(stage='decoding')
                    receipt = {**row, 'video': str(video), 'verified_md5': md5,'decode_policy':FRAME_POLICY,
                               'frames': decode_bounded(video, folder, guard), 'storage': 'expansion_directory'}
                    validate_receipt(receipt, row, destination, guard)
                    atomic_json(receipt_path, receipt)
                completed.append(receipt)
                manifest = {**plan, 'plan_sha256': plan_sha, 'rows': completed, 'complete': False,
                            'frame_indices':None,'preferred_frame_indices':FRAME_INDICES,'frame_policy':FRAME_POLICY,
                            'dish_id_count': len(completed), 'decoded_frames': len(completed) * 3}
                atomic_json(destination / 'manifest.json', manifest)
                report(stage='dish_complete', dishes_done=len(completed), frames=len(completed) * 3)
            manifest.update(complete=True, completed_at=now(),
                pilot_reference_dishes=sum(r['storage'] == 'verified_pilot_reference' for r in completed))
            atomic_json(destination / 'manifest.json', manifest)
            atomic_json(ROOT / 'results/video_expansion_v1.json', {
                'manifest': str(destination / 'manifest.json'), 'sha256': file_digest(destination / 'manifest.json'),
                'plan_sha256': plan_sha, 'dishes': len(completed), 'frames': len(completed) * 3,
                'planned_video_bytes': plan['planned_video_bytes'],
                'pilot_reference_dishes': manifest['pilot_reference_dishes'], 'included_in_training': False})
            report(stage='complete', dishes_done=len(completed), frames=len(completed) * 3)
        except BaseException as exc:
            failure = {'stage': 'paused' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
                       'error_type': type(exc).__name__, 'dish_id': state.get('dish_id'),
                       'last_stage': state.get('stage'), 'updated_at': now()}
            if not isinstance(exc, requests.RequestException):
                failure['reason'] = str(exc)
            atomic_json(destination / 'failure.json', failure)
            state.update(failure)
            atomic_json(state_path, state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default=str(DESTINATION))
    parser.add_argument('--plan-only', action='store_true', help='Freeze official metadata without video downloads')
    parser.add_argument('--retry-failed', action='store_true', help='Resume diagnosed failure within original deadline')
    parser.add_argument('--decode-video', help=argparse.SUPPRESS)
    parser.add_argument('--decode-folder', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.decode_video or args.decode_folder:
        if not args.decode_video or not args.decode_folder:
            parser.error('Both internal decode arguments are required')
        video, folder = scoped(args.decode_video, DESTINATION), scoped(args.decode_folder, DESTINATION)
        if video.parent != folder or not re.fullmatch(r'dish_\d+', folder.name):
            raise ValueError('Invalid internal decode paths')
        check_stop(DESTINATION)
        atomic_json(folder / 'decoded_frames.json', extract_expansion_frames(video, folder))
    else:
        run(args.destination, args.plan_only, args.retry_failed)
