"""Bounded, resumable local Capsicum download -> validation -> training job.

Source: https://zenodo.org/records/6340415 (CC BY 4.0).
State/logs are local; no credential use, uploads, or changes to the demo.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
import tarfile
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'deepNIR_capsicum'
NAME = 'capsicums_pix2pixHD_8_1_1.tar.gz'
URL = f'https://zenodo.org/api/records/6340415/files/{NAME}/content'
SIZE = 3256378152
MD5 = '1d8ad73bcfe917f23806fd3417dc5d12'
MIN_FREE = 35 * 1024**3


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus may briefly deny FILE_SHARE_DELETE. Retry only
    # this atomic rename; never delete the last valid status or hide failure.
    for attempt in range(7):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * 2**attempt)


def check_space(path, min_free_bytes=MIN_FREE):
    if shutil.disk_usage(path).free < min_free_bytes:
        raise RuntimeError('Disk free space below configured safety reserve')


def check_stop(path):
    if (Path(path) / 'STOP').exists():
        raise InterruptedError('STOP file present; job paused without deleting data')


def file_digest(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, expected_size=SIZE, expected_md5=MD5):
    if Path(path).stat().st_size != expected_size:
        raise ValueError('Archive size mismatch; extraction prohibited')
    actual = file_digest(path, 'md5')
    if actual != expected_md5:
        raise ValueError(f'Archive MD5 mismatch: {actual}; original retained')
    return actual


def resume_download(url, dest, expected_size, report, *, min_free_bytes=MIN_FREE,
                    retry_delay=15, max_seconds=12 * 3600, max_no_progress=12):
    """Persist successful bytes; validate Range responses before appending.

    Retry only transient network/HTTP failures. Each read has a timeout; each
    connection is recycled after 120 seconds. No rollback on partial transfers.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial = dest.stat().st_size if dest.exists() else 0
    attempts = 0
    no_progress = 0
    last_report = 0.0
    while True:
        check_stop(dest.parent)
        check_space(dest.parent, min_free_bytes)
        offset = dest.stat().st_size if dest.exists() else 0
        if offset > expected_size:
            raise ValueError('Existing file larger than official size; retained for inspection')
        if offset == expected_size:
            report(stage='downloaded', bytes=offset, total_bytes=expected_size, attempts=attempts)
            return
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('Download deadline exceeded; partial data retained for resume')
        attempts += 1
        report(stage='downloading', bytes=offset, total_bytes=expected_size, attempts=attempts)
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'YituZhika-dataset-recovery/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        error = None
        wait_seconds = retry_delay
        attempt_started = time.monotonic()
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 20)) as response:
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    delay = response.headers.get('Retry-After', '')
                    if delay.isdigit():
                        wait_seconds = max(wait_seconds, min(300, int(delay)))
                    raise requests.ConnectionError(f'Transient HTTP {response.status_code}')
                if response.status_code not in (200, 206):
                    raise ValueError(f'HTTP {response.status_code}: access/URL needs review')
                if offset or response.status_code == 206:
                    value = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
                    if response.status_code != 206 or not match:
                        raise ValueError('Server ignored Range; refusing to append wrong bytes')
                    first, last, total = map(int, match.groups())
                    if first != offset or total != expected_size or last >= total or last < first:
                        raise ValueError(f'Unexpected Content-Range: {value}')
                elif int(response.headers.get('Content-Length', '-1')) != expected_size:
                    raise ValueError('Unexpected server file size')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise ValueError('Compressed HTTP body would invalidate byte offsets')
                with dest.open('ab') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        if f.tell() + len(chunk) > expected_size:
                            raise ValueError('Response exceeds official file size')
                        f.write(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            f.flush()
                            check_stop(dest.parent)
                            check_space(dest.parent, min_free_bytes)
                            report(stage='downloading', bytes=f.tell(), total_bytes=expected_size,
                                   attempts=attempts, bytes_per_second=round((f.tell()-initial)/max(now-started, 0.01)))
                            last_report = now
                        if now - attempt_started > 120:
                            break
        except requests.RequestException as exc:
            # Exception text can include proxy credentials. Keep only its class.
            error = type(exc).__name__
        current = dest.stat().st_size if dest.exists() else 0
        no_progress = no_progress + 1 if current == offset else 0
        if current == expected_size:
            continue
        if no_progress >= max_no_progress:
            raise RuntimeError(f'{no_progress} attempts without progress; resume remains possible')
        report(stage='retrying' if error else 'reconnecting', bytes=current,
               total_bytes=expected_size, attempts=attempts, last_network_error=error)
        # Successful bounded transfers reconnect immediately. Failed requests back off.
        if error:
            deadline = time.monotonic() + min(300, wait_seconds * max(1, no_progress))
            while time.monotonic() < deadline:
                check_stop(dest.parent)
                time.sleep(min(5, max(0, deadline-time.monotonic())))


def extract_archive(archive, destination, report, *, min_free_bytes=MIN_FREE):
    """Preflight all members, reject links/traversal/oversize, then extract."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as t:
        members = t.getmembers()
        total = sum(m.size for m in members if m.isfile())
        if total > 8 * 1024**3 or len(members) > 15000:
            raise ValueError('Archive exceeds extraction limits')
        check_space(destination, min_free_bytes + total)
        for m in members:
            p = PurePosixPath(m.name)
            if (p.is_absolute() or '..' in p.parts or '\\' in m.name or
                    ':' in m.name or PureWindowsPath(m.name).drive or
                    not (m.isfile() or m.isdir())):
                raise ValueError(f'Unsafe tar member: {m.name}')
            target = (destination / m.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Tar member escapes destination')
        for index, m in enumerate(members):
            check_stop(Path(archive).parent)
            target = destination / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024**2)
            if index % 100 == 0:
                report(stage='extracting', files_done=index, total_files=len(members))
    return destination


def validate_dataset(extracted, report, expected_pairs=1615):
    """Use explicit official splits and exact names; never index-zip or test-as-val."""
    import numpy as np
    from PIL import Image

    roots = [p.parent for p in Path(extracted).rglob('train_A') if p.is_dir()]
    if len(roots) != 1:
        raise ValueError(f'Expected exactly one train_A root, got {roots}')
    root = roots[0]
    rows = []
    seen_ids = set()
    seen_hashes = {'A': {}, 'B': {}}
    orientations = set()
    counts = {}
    for split in ('train', 'val', 'test'):
        sides = {}
        for side in ('A', 'B'):
            folder = root / f'{split}_{side}'
            if not folder.is_dir():
                raise ValueError(f'Missing explicit {split}_{side}; no split fallback allowed')
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
            sides[side] = {p.stem: p for p in files}
            if len(sides[side]) != len(files):
                raise ValueError('Duplicate image stems')
        if not sides['A'] or sides['A'].keys() != sides['B'].keys():
            raise ValueError(f'{split}: RGB/NIR filenames do not pair exactly')
        counts[split] = len(sides['A'])
        for stem in sides['A']:
            # Drop the export sequence index, preserving the original camera/frame ID.
            sample_id = re.sub(r'-\d{4}$', '', stem)
            if sample_id in seen_ids:
                raise ValueError(f'Duplicate physical frame ID across dataset: {sample_id}')
            seen_ids.add(sample_id)
            arrays = {}
            for side in ('A', 'B'):
                path = sides[side][stem]
                with Image.open(path) as img:
                    img.load()
                    arrays[side] = np.asarray(img.convert('RGB'))
                digest = hashlib.sha256(arrays[side].tobytes()).hexdigest()
                prior = seen_hashes[side].get(digest)
                if prior and prior != split:
                    raise ValueError(f'Image content leakage: {prior} -> {split}')
                seen_hashes[side][digest] = split
            if arrays['A'].shape != arrays['B'].shape:
                raise ValueError('Pair geometry mismatch')
            gray = {s: bool(np.array_equal(a[:,:,0], a[:,:,1]) and np.array_equal(a[:,:,0], a[:,:,2]))
                    for s, a in arrays.items()}
            if gray['A'] == gray['B']:
                raise ValueError('Cannot unambiguously identify RGB vs NIR')
            nir_side = 'A' if gray['A'] else 'B'
            rgb_side = 'B' if gray['A'] else 'A'
            orientations.add(nir_side)
            rows.append({'id': sample_id, 'split': split,
                         'rgb': str(sides[rgb_side][stem].relative_to(root)),
                         'nir': str(sides[nir_side][stem].relative_to(root))})
            if len(rows) % 50 == 0:
                report(stage='validating', pairs_checked=len(rows))
    if len(orientations) != 1 or len(rows) != expected_pairs:
        raise ValueError(f'Unexpected dataset size/orientation: {len(rows)}, {orientations}')
    manifest = {'source': 'https://zenodo.org/records/6340415', 'license': 'CC-BY-4.0',
                'archive_md5': MD5, 'root': str(root), 'counts': counts, 'pairs': rows,
                'protocol': 'Official 8:1:1 splits; exact frame/content leakage checks; test excluded from selection.',
                'limitations': 'Farm-domain auxiliary RGB-to-NIR study; not a food-calorie benchmark. Near-duplicate capture correlation is not ruled out.'}
    return manifest


@contextmanager
def job_lock(path):
    """OS lock releases on crash; a stale lock file is harmless."""
    import msvcrt
    f = Path(path).open('a+b')
    if f.tell() == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        raise RuntimeError('Another Capsicum job already holds the lock')
    try:
        yield
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


def run(args):
    DATA.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'source': URL, 'official_bytes': SIZE, 'official_md5': MD5}
    def report(**values):
        state.update(values)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(DATA / 'job_status.json', state)
        print(json.dumps(values, ensure_ascii=False), flush=True)
    with job_lock(DATA / 'job.lock'):
        try:
            report(stage='starting')
            archive = DATA / NAME
            resume_download(URL, archive, SIZE, report)
            report(stage='checksum')
            digest = verify_archive(archive)
            report(stage='verified', md5=digest)
            extracted = DATA / 'extracted'
            extract_archive(archive, extracted, report)
            manifest = validate_dataset(extracted, report)
            atomic_json(DATA / 'manifest.json', manifest)
            report(stage='validated', counts=manifest['counts'])
            if args.download_only:
                report(stage='data_ready')
                return
            check_space(DATA)
            cmd = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'src/training/train_capsicum.py'),
                   '--manifest', str(DATA / 'manifest.json'), '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size)]
            train_log = DATA / 'training_console.log'
            if getattr(args, 'allow_amp_policy_migration', False):
                cmd.append('--allow-amp-policy-migration')
            with train_log.open('a', encoding='utf-8') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                report(stage='smoke_then_training', training_pid=child.pid, training_log=str(train_log))
                try:
                    while child.poll() is None:
                        check_stop(DATA)
                        check_space(DATA)
                        report(training_pid=child.pid)
                        time.sleep(10)
                except BaseException:
                    child.terminate()
                    child.wait(timeout=30)
                    raise
                if child.returncode:
                    raise RuntimeError(f'Training failed, exit={child.returncode}; see training_console.log')
            report(stage='complete', training_exit_code=0)
        except BaseException as exc:
            report(stage='paused' if isinstance(exc, InterruptedError) else 'failed',
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--allow-amp-policy-migration', action='store_true')
    run(p.parse_args())
