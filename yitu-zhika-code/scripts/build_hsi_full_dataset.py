"""把队友的全量 HSIFoodIngr-64（h5）转换为本项目的 RGBN memmap 数据，用于阶段一复现训练。

关键点：
- RGB：直接取 h5 的 uint8，转为 [0,1]；
- NIR：h5 的 uint16 相对 RGB **旋转了 90°**，需 `np.rot90(nir, 3)` 对齐；再用 144 张重叠扫描标定出的
  仿射映射转成我们的固定缩放口径：`our_nir = slope * u16_rot + intercept`，截断 [0,1]；
- 输出 memmap：`rgb_<split>.npy` (uint8, N×256×256×3)、`nir_<split>.npy` (float16, N×256×256)，
  加 `manifest.json`（逐样本 id/split/下标 + 来源 h5 哈希 + 标定参数 + 训练集统计）。

用法（必须用有 h5py 的解释器）：
  C:\\Users\\user\\anaconda3\\python.exe scripts/build_hsi_full_dataset.py
"""
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r'C:\Users\user\Desktop\新建文件夹\model_v2\data\HSIFoodIngr-64')
OUT = ROOT / 'data/hsi_full_v3'
CAL = json.loads((ROOT / 'artifacts/v2_nir_calibration_rot.json').read_text(encoding='utf-8'))
SLOPE, INTERCEPT = CAL['slope'], CAL['intercept']


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, source_hashes = [], {}
    stats = {}
    for split in ('train', 'val', 'test'):
        files = sorted(glob.glob(str(SRC / split / '*.h5')))
        ids = []
        for f in files:
            with h5py.File(f, 'r') as h:
                ids += [n.decode() if isinstance(n, bytes) else str(n) for n in h['names'][:]]
            source_hashes[str(Path(f).relative_to(SRC))] = sha256(f)
        n = len(ids)
        rgb_mm = np.lib.format.open_memmap(OUT / f'rgb_{split}.npy', mode='w+',
                                           dtype=np.uint8, shape=(n, 256, 256, 3))
        nir_mm = np.lib.format.open_memmap(OUT / f'nir_{split}.npy', mode='w+',
                                           dtype=np.float16, shape=(n, 256, 256))
        i = 0
        raw_min, raw_max = 65535, 0
        for f in files:
            with h5py.File(f, 'r') as h:
                names = [x.decode() if isinstance(x, bytes) else str(x) for x in h['names'][:]]
                for j, name in enumerate(names):
                    rgb = np.asarray(h['rgb'][j], dtype=np.uint8)
                    nir16 = np.asarray(h['nir'][j], dtype=np.uint16)
                    raw_min = min(raw_min, int(nir16.min()))
                    raw_max = max(raw_max, int(nir16.max()))
                    nir01 = np.clip(SLOPE * np.rot90(nir16.astype(np.float32), 3) + INTERCEPT, 0, 1)
                    rgb_mm[i] = rgb
                    nir_mm[i] = nir01.astype(np.float16)
                    rows.append({'id': name, 'split': split, 'index': i})
                    i += 1
            print(f'  {split}: {Path(f).name} 累计 {i}/{n}', flush=True)
        rgb_mm.flush(); nir_mm.flush()
        del rgb_mm, nir_mm
        stats[split] = {'n': n, 'nir_raw_u16_min': raw_min, 'nir_raw_u16_max': raw_max}
        print(f'{split}: {n} 样本 -> rgb_{split}.npy / nir_{split}.npy', flush=True)

    # 训练集 NIR 统计（写入清单，供溯源）
    nir_train = np.load(OUT / 'nir_train.npy', mmap_mode='r')
    sample = np.asarray(nir_train[:: max(1, len(nir_train) // 200)]).astype(np.float32)
    train_stats = {'nir01_mean': float(sample.mean()), 'nir01_std': float(sample.std()),
                   'nir01_p0.5': float(np.percentile(sample, 0.5)),
                   'nir01_p99.5': float(np.percentile(sample, 99.5))}

    manifest = {
        'protocol': 'hsi_full_v3_rgb859_from_teammate_h5_aligned',
        'source_dir': str(SRC),
        'source_h5_sha256': source_hashes,
        'conversion': {
            'rotation': 'np.rot90(nir_u16, 3)  # 对齐到数据集自带 RGB PNG 朝向',
            'nir_affine': {'slope': SLOPE, 'intercept': INTERCEPT,
                           'source': 'artifacts/v2_nir_calibration_rot.json（144 张重叠扫描标定, R²=0.9990）'},
            'rgb': 'h5 uint8 / 255 -> [0,1]',
            'output': 'rgb_<split>.npy uint8 [0,255]; nir_<split>.npy float16 [0,1]',
        },
        'counts': {k: v['n'] for k, v in stats.items()},
        'split_stats': stats,
        'train_nir_stats': train_stats,
        'rows': rows,
        'limitations': [
            'NIR 目标为 860 nm 单波段，经训练集统计缩放（与本地 144 扫描版同口径）',
            '划分沿用队友清洗后的按拍摄日/文件划分；不是官方 RGB→NIR 官方划分',
            '本清单用于阶段一复现训练，不参与阶段二 Nutrition5k 任务',
        ],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print('清单:', OUT / 'manifest.json')
    print('划分:', manifest['counts'])
    print('训练集 NIR 统计:', train_stats)


if __name__ == '__main__':
    main()
