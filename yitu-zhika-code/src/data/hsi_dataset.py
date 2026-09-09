"""
HSIDataset: RGB+NIR 配对数据集 (Phase1)
========================================
从 HSIFoodIngr-64 的 ENVI .hdr/.raw 文件读取 204 波段高光谱数据，
提取 RGB(640/550/460nm) + NIR(860nm) 四通道，返回配对张量。

使用方法:
    from src.data.hsi_dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(data_root="...", batch_size=4)
    for rgb, nir in train_loader:
        # rgb: [B, 3, H, W]  nir: [B, 1, H, W]  值域 [-1, 1]
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────── ENVI .hdr 解析 ────────────────────

def parse_envi_hdr(hdr_path: str) -> dict:
    """解析 ENVI .hdr 头文件，返回元信息字典。"""
    info = {}
    with open(hdr_path, 'r') as f:
        text = f.read()
    # 先提取所有花括号多行值（如 wavelength, default bands 等）
    brace_match = re.finditer(r'([\w\s]+?)\s*=\s*\{([^}]*)\}', text, re.DOTALL)
    for m in brace_match:
        k = m.group(1).strip().lower()
        v = m.group(2).strip()
        info[k] = v
    # 再提取单行键值对（跳过已匹配的花括号值）
    for m in re.finditer(r'([\w][\w\s]*?)\s*=\s*(\{[^}]*\}|[^\n]+)', text):
        k = m.group(1).strip().lower()
        v = m.group(2).strip()
        if k not in info:  # 不覆盖已解析的花括号值
            info[k] = v
    # 整数字段安全转换
    for key in ['samples', 'lines', 'bands']:
        if key in info:
            try:
                info[key] = int(str(info[key]).strip())
            except ValueError:
                pass
    # data type 可能是多行花括号，取第一个值
    if 'data type' in info:
        dt_str = str(info['data type']).strip()
        try:
            info['data type'] = int(dt_str)
        except ValueError:
            # 多行格式如 "70, 53, 19"，取第一个
            first = dt_str.lstrip('{').split(',')[0].strip()
            try:
                info['data type'] = int(first)
            except ValueError:
                info['data type'] = 4  # 默认 float32
    # wavelength
    wl_match = re.search(r'wavelength\s*=\s*\{([^}]*)\}', text, re.DOTALL)
    if wl_match:
        wl_str = wl_match.group(1)
        info['wavelength'] = [float(x.strip()) for x in wl_str.split(',') if x.strip()]
    return info


def read_envi_raw(raw_path: str, hdr_info: dict) -> np.ndarray:
    """读取 ENVI .raw 二进制数据，返回 (bands, H, W) float64 数组。"""
    samples = hdr_info['samples']
    lines = hdr_info['lines']
    bands = hdr_info['bands']
    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32}
    dt = dtype_map.get(hdr_info.get('data type', 4), np.float32)
    interleave = hdr_info.get('interleave', 'bsq').lower()

    data = np.fromfile(raw_path, dtype=dt)
    expected = samples * lines * bands
    if len(data) < expected:
        raise ValueError(f"Raw file too short: {len(data)} < {expected}")

    if interleave == 'bsq':
        data = data[:expected].reshape(bands, lines, samples)
    elif interleave == 'bil':
        data = data[:expected].reshape(lines, bands, samples).transpose(1, 0, 2)
    elif interleave == 'bip':
        data = data[:expected].reshape(lines, samples, bands).transpose(2, 0, 1)
    else:
        data = data[:expected].reshape(bands, lines, samples)

    return data.astype(np.float64)


def select_band_indices(wavelengths: list, targets: dict = None) -> dict:
    """根据波长列表找最近波段索引。"""
    if targets is None:
        targets = {'red': 640.0, 'green': 550.0, 'blue': 460.0, 'nir': 860.0}
    wl = np.array(wavelengths)
    result = {}
    for name, target_nm in targets.items():
        idx = int(np.argmin(np.abs(wl - target_nm)))
        result[name] = idx
        result[f'{name}_nm'] = wl[idx]
    return result


def percentile_normalize(band_data: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """逐波段 percentile clip 后归一化到 [0, 1]。"""
    lo = np.percentile(band_data, pmin)
    hi = np.percentile(band_data, pmax)
    if hi - lo < 1e-8:
        return np.zeros_like(band_data)
    return np.clip((band_data - lo) / (hi - lo), 0.0, 1.0)


# ──────────────────── Dataset ────────────────────

class HSIFoodIngrDataset(Dataset):
    """HSIFoodIngr-64 RGB+NIR 配对数据集。

    Returns:
        rgb: [3, H, W]  float32, 值域 [-1, 1]
        nir: [1, H, W]  float32, 值域 [-1, 1]
    """

    TARGET_WAVELENGTHS = {'red': 640.0, 'green': 550.0, 'blue': 460.0, 'nir': 860.0}

    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 256, val_ratio: float = 0.15):
        super().__init__()
        self.img_size = img_size
        self.split = split

        self.samples = []
        for subdir in sorted(os.listdir(data_root)):
            sub_path = os.path.join(data_root, subdir)
            if not os.path.isdir(sub_path):
                continue
            for hdr_file in sorted(glob.glob(os.path.join(sub_path, '*.hdr'))):
                # 兼容 .raw 和 .dat 两种扩展名
                raw_file = hdr_file.replace('.hdr', '.raw')
                if not os.path.exists(raw_file):
                    raw_file = hdr_file.replace('.hdr', '.dat')
                if os.path.exists(raw_file):
                    self.samples.append((hdr_file, raw_file))

        n = len(self.samples)
        n_val = int(n * val_ratio)
        if split == 'train':
            self.samples = self.samples[:n - n_val]
        elif split == 'val':
            self.samples = self.samples[n - n_val:]

        print(f"[HSIDataset] {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdr_path, raw_path = self.samples[idx]
        hdr_info = parse_envi_hdr(hdr_path)
        wavelengths = hdr_info.get('wavelength', None)
        hsi = read_envi_raw(raw_path, hdr_info)

        if wavelengths and len(wavelengths) == hsi.shape[0]:
            bands = select_band_indices(wavelengths, self.TARGET_WAVELENGTHS)
            r = percentile_normalize(hsi[bands['red']])
            g = percentile_normalize(hsi[bands['green']])
            b = percentile_normalize(hsi[bands['blue']])
            nir = percentile_normalize(hsi[bands['nir']])
        else:
            n_bands = hsi.shape[0]
            r = percentile_normalize(hsi[n_bands * 2 // 3])
            g = percentile_normalize(hsi[n_bands // 2])
            b = percentile_normalize(hsi[n_bands // 4])
            nir = percentile_normalize(hsi[int(n_bands * 0.75)])

        from PIL import Image
        def resize_band(arr):
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            return np.array(img).astype(np.float32) / 255.0

        r = resize_band(r)
        g = resize_band(g)
        b = resize_band(b)
        nir = resize_band(nir)

        rgb = np.stack([r, g, b], axis=0)
        nir_arr = nir[np.newaxis, :, :]
        rgb_tensor = torch.from_numpy(rgb * 2.0 - 1.0)
        nir_tensor = torch.from_numpy(nir_arr * 2.0 - 1.0)

        return rgb_tensor, nir_tensor


def build_dataloaders(data_root: str, batch_size: int = 4,
                      num_workers: int = 2, img_size: int = 256):
    train_ds = HSIFoodIngrDataset(data_root, split='train', img_size=img_size)
    val_ds = HSIFoodIngrDataset(data_root, split='val', img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == '__main__':
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\data\HSIFoodIngr-64"
    print("=== HSIDataset Test ===")
    ds = HSIFoodIngrDataset(data_root, split='train', img_size=256)
    if len(ds) > 0:
        rgb, nir = ds[0]
        print(f"  rgb: {rgb.shape}, range [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"  nir: {nir.shape}, range [{nir.min():.2f}, {nir.max():.2f}]")
    else:
        print("  No samples found!")
