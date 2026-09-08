"""
从真实 HSIFoodIngr-64 ENVI 格式数据提取 RGB + NIR 配对图像。
- RGB: 从 .png 文件直接读取
- NIR: 从 .dat 高光谱立方体中提取近红外波段（~850nm 附近）
"""
import os
import sys
import numpy as np
from PIL import Image
import struct
import re
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

HSI_ROOT = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\data\HSIFoodIngr-64"
OUT_RGB = "data/HSIFoodIngr-64_real/RGB"
OUT_NIR = "data/HSIFoodIngr-64_real/NIR"
os.makedirs(OUT_RGB, exist_ok=True)
os.makedirs(OUT_NIR, exist_ok=True)

def parse_hdr(hdr_path):
    """解析 ENVI hdr 文件，返回 samples, lines, bands, wavelengths, interleave, data_type"""
    info = {"samples": 0, "lines": 0, "bands": 0, "interleave": "bil", "data_type": 4, "wavelength": []}
    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = re.search(r'samples\s*=\s*(\d+)', text)
    if m: info["samples"] = int(m.group(1))
    m = re.search(r'lines\s*=\s*(\d+)', text)
    if m: info["lines"] = int(m.group(1))
    m = re.search(r'bands\s*=\s*(\d+)', text)
    if m: info["bands"] = int(m.group(1))
    m = re.search(r'interleave\s*=\s*(\w+)', text)
    if m: info["interleave"] = m.group(1).lower()
    m = re.search(r'data\s+type\s*=\s*(\d+)', text)
    if m: info["data_type"] = int(m.group(1))
    # 提取波长
    wl_match = re.search(r'wavelength\s*=\s*\{([^}]+)\}', text)
    if wl_match:
        wl_str = wl_match.group(1)
        wls = [float(x.strip()) for x in wl_str.split(',') if x.strip()]
        info["wavelength"] = wls
    return info

def read_envi_band(dat_path, hdr_info, band_idx):
    """从 ENVI BIL 格式 .dat 文件中读取指定波段"""
    samples = hdr_info["samples"]
    lines = hdr_info["lines"]
    bands = hdr_info["bands"]
    dtype = np.float32 if hdr_info["data_type"] == 4 else np.float32

    interleave = hdr_info["interleave"]
    # BIL: Band Interleaved by Line - 每行存储所有波段
    # 数据排列: [line0_band0, line0_band1, ..., line0_bandN, line1_band0, ...]
    data = np.fromfile(dat_path, dtype=dtype)

    expected_size = samples * lines * bands
    if len(data) < expected_size:
        data = data[:expected_size]

    if interleave == "bil":
        # BIL: [line, band, sample] -> reshape
        data = data.reshape(lines, bands, samples)
        band_data = data[:, band_idx, :]  # [lines, samples]
    elif interleave == "bip":
        # BIP: [line, sample, band]
        data = data.reshape(lines, samples, bands)
        band_data = data[:, :, band_idx]  # [lines, samples]
    elif interleave == "bsq":
        # BSQ: [band, line, sample]
        data = data.reshape(bands, lines, samples)
        band_data = data[band_idx, :, :]
    else:
        data = data.reshape(lines, bands, samples)
        band_data = data[:, band_idx, :]

    return band_data

def find_nir_band(wavelengths, target_nm=850):
    """找到最接近目标波长的波段索引"""
    if not wavelengths:
        return None
    diffs = [abs(w - target_nm) for w in wavelengths]
    return diffs.index(min(diffs))

def process_sample(dat_path, hdr_path, png_path, out_name):
    """处理一个样本，提取 RGB + NIR"""
    hdr_info = parse_hdr(hdr_path)

    # RGB: 从 PNG 读取
    rgb_img = Image.open(png_path).convert("RGB")
    rgb_img = rgb_img.resize((256, 256), Image.LANCZOS)
    rgb_path = os.path.join(OUT_RGB, out_name + ".png")
    rgb_img.save(rgb_path)

    # NIR: 从 .dat 提取近红外波段
    nir_band_idx = find_nir_band(hdr_info["wavelength"], 850)
    if nir_band_idx is None:
        # 如果没有波长信息，取中间偏后的波段
        nir_band_idx = hdr_info["bands"] * 3 // 4

    nir_data = read_envi_band(dat_path, hdr_info, nir_band_idx)

    # 归一化到 0-255
    nir_min, nir_max = nir_data.min(), nir_data.max()
    if nir_max > nir_min:
        nir_norm = (nir_data - nir_min) / (nir_max - nir_min) * 255
    else:
        nir_norm = np.zeros_like(nir_data)
    nir_uint8 = nir_norm.astype(np.uint8)
    nir_img = Image.fromarray(nir_uint8, mode='L')
    nir_img = nir_img.resize((256, 256), Image.LANCZOS)
    nir_path = os.path.join(OUT_NIR, out_name + ".png")
    nir_img.save(nir_path)

    return True

def main():
    dirs = sorted(os.listdir(HSI_ROOT))
    print(f"找到 {len(dirs)} 个子目录")

    processed = 0
    skipped = 0

    for dname in dirs:
        dpath = os.path.join(HSI_ROOT, dname)
        if not os.path.isdir(dpath):
            continue

        files = os.listdir(dpath)
        # 按 REFLECTANCE 编号分组
        reflectance_ids = set()
        for f in files:
            if f.startswith("REFLECTANCE_") and f.endswith(".dat"):
                rid = f.replace("REFLECTANCE_", "").replace(".dat", "")
                reflectance_ids.add(rid)

        for rid in sorted(reflectance_ids):
            dat_path = os.path.join(dpath, f"REFLECTANCE_{rid}.dat")
            hdr_path = os.path.join(dpath, f"REFLECTANCE_{rid}.hdr")
            png_path = os.path.join(dpath, f"REFLECTANCE_{rid}.png")

            if not all(os.path.exists(p) for p in [dat_path, hdr_path, png_path]):
                skipped += 1
                continue

            out_name = f"{dname}_{rid}"
            try:
                process_sample(dat_path, hdr_path, png_path, out_name)
                processed += 1
                if processed % 10 == 0:
                    print(f"  已处理 {processed} 个样本...")
            except Exception as e:
                print(f"  [SKIP] {out_name}: {e}")
                skipped += 1

    print(f"\n=== 提取完成 ===")
    print(f"成功: {processed}")
    print(f"跳过: {skipped}")
    print(f"RGB → {OUT_RGB}/")
    print(f"NIR → {OUT_NIR}/")

if __name__ == "__main__":
    main()
