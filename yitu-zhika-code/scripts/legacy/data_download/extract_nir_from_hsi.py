"""从 HSIFoodIngr-64 .dat 文件提取真实 NIR 波段，生成 RGB+NIR 配对数据集。

Specim IQ 高光谱相机：
- 204 波段，397.32 - 1003.58 nm
- BIL interleave, float32, 512×512
- 选取 ~800 nm 波段（波段 150, 838.14 nm）作为 NIR 通道
- 选取波段 70/53/19（RGB 默认波段）作为 RGB 通道

输出：data/hsi_nir_pairs/
  trainA/  - RGB 图像 (3 通道 PNG)
  trainB/  - NIR 图像 (1 通道 PNG, 伪彩或灰度)
  valA/, valB/
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# 添加项目根目录到 sys.path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ═════════════════════ 配置 ═════════════════════
HSI_ROOT = ROOT / "data" / "HSIFoodIngr-64"
OUTPUT_ROOT = ROOT / "data" / "hsi_nir_pairs"

# Specim IQ 波段选择
RGB_BANDS = [70, 53, 19]      # hdr default bands, 约 610/530/455 nm (R/G/B)
NIR_BAND = 150                 # 838.14 nm，真实 NIR 波段
IMG_SIZE = 256                 # 缩放到 256×256（原 512×512 太大）
VAL_RATIO = 0.15               # 15% 验证集
SEED = 42


def read_envi_dat(dat_path: str, hdr_path: str) -> np.ndarray:
    """读取 ENVI BIL 格式 .dat 文件，返回 [bands, height, width] 的 float32 数组。"""
    # 解析 hdr
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        hdr = f.read()

    def parse_int(key):
        m = np.nan
        for line in hdr.split("\n"):
            if line.strip().startswith(key):
                val = line.split("=")[-1].strip()
                try:
                    m = int(float(val))
                except ValueError:
                    pass
        return m

    samples = parse_int("samples")    # width
    lines = parse_int("lines")        # height
    bands = parse_int("bands")         # bands
    data_type = parse_int("data type")  # 4 = float32

    if data_type != 4:
        raise ValueError(f"不支持的 data type: {data_type}, 只支持 4 (float32)")

    # 读取二进制
    raw = np.fromfile(dat_path, dtype=np.float32)
    expected = samples * lines * bands
    if raw.size != expected:
        # 可能是 little-endian/big-endian 差异
        raw = np.fromfile(dat_path, dtype=np.float32.newbyteorder("<"))
        if raw.size != expected:
            raise ValueError(
                f"数据大小不匹配: {raw.size} vs 预期 {expected}"
            )

    # BIL interleave: [lines, bands, samples] → 重排为 [bands, lines, samples]
    arr = raw.reshape((lines, bands, samples))
    # BIL: 行优先，每行存所有波段 → 转 [bands, height, width]
    arr = arr.transpose(1, 0, 2)
    return arr  # [bands, height, width]


def normalize_to_uint8(arr: np.ndarray, percentile=99.5) -> np.ndarray:
    """把 float32 反射率归一化到 0-255 uint8。"""
    p_low = np.percentile(arr, 0.5)
    p_high = np.percentile(arr, percentile)
    arr = np.clip(arr, p_low, p_high)
    arr = (arr - p_low) / (p_high - p_low + 1e-8) * 255.0
    return arr.astype(np.uint8)


def extract_and_save(dat_path, hdr_path, out_rgb_path, out_nir_path):
    """提取 RGB + NIR 波段并保存为 PNG。"""
    try:
        arr = read_envi_dat(str(dat_path), str(hdr_path))
    except Exception as e:
        print(f"  [WARN] 跳过 {dat_path.name}: {e}")
        return False

    # 提取 RGB 波段
    rgb = arr[RGB_BANDS, :, :]  # [3, H, W]
    # 提取 NIR 波段
    nir = arr[NIR_BAND, :, :]   # [H, W]

    # 归一化
    rgb_uint8 = np.stack([normalize_to_uint8(rgb[i]) for i in range(3)], axis=2)  # [H, W, 3]
    nir_uint8 = normalize_to_uint8(nir)  # [H, W]

    # 缩放到 256×256
    img_rgb = Image.fromarray(rgb_uint8, mode="RGB")
    img_nir = Image.fromarray(nir_uint8, mode="L")
    img_rgb = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_nir = img_nir.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    img_rgb.save(out_rgb_path)
    img_nir.save(out_nir_path)
    return True


def main():
    print("=" * 60)
    print("从 HSIFoodIngr-64 提取真实 NIR 波段（838 nm）")
    print("=" * 60)

    # 收集所有 .dat 文件
    dat_files = sorted(HSI_ROOT.rglob("REFLECTANCE_*.dat"))
    print(f"找到 {len(dat_files)} 个 .dat 文件")

    # 创建输出目录
    for split in ["trainA", "trainB", "valA", "valB"]:
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    # 随机划分
    np.random.seed(SEED)
    indices = np.random.permutation(len(dat_files))
    val_count = int(len(dat_files) * VAL_RATIO)
    val_indices = set(indices[:val_count])

    success = 0
    failed = 0
    for i, dat_path in enumerate(dat_files):
        hdr_path = dat_path.with_suffix(".hdr")
        if not hdr_path.exists():
            print(f"  [SKIP] {dat_path.name}: 无 .hdr 文件")
            failed += 1
            continue

        # 文件名（去掉 REFLECTANCE_ 前缀和路径）
        stem = dat_path.stem  # e.g. REFLECTANCE_218
        parent_name = dat_path.parent.name  # e.g. HSIFoodIngr-64_data_1
        name = f"{parent_name}_{stem}.png"

        if i in val_indices:
            out_rgb = OUTPUT_ROOT / "valA" / name
            out_nir = OUTPUT_ROOT / "valB" / name
        else:
            out_rgb = OUTPUT_ROOT / "trainA" / name
            out_nir = OUTPUT_ROOT / "trainB" / name

        ok = extract_and_save(dat_path, hdr_path, out_rgb, out_nir)
        if ok:
            success += 1
            if success % 10 == 0:
                print(f"  已处理 {success}/{len(dat_files)}")
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"完成！成功: {success}, 失败: {failed}")
    print(f"训练集: {len(dat_files) - val_count} 对")
    print(f"验证集: {val_count} 对")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 60)

    # 验证输出
    train_a = len(list((OUTPUT_ROOT / "trainA").glob("*.png")))
    train_b = len(list((OUTPUT_ROOT / "trainB").glob("*.png")))
    val_a = len(list((OUTPUT_ROOT / "valA").glob("*.png")))
    val_b = len(list((OUTPUT_ROOT / "valB").glob("*.png")))
    print(f"验证: trainA={train_a}, trainB={train_b}, valA={val_a}, valB={val_b}")


if __name__ == "__main__":
    main()
