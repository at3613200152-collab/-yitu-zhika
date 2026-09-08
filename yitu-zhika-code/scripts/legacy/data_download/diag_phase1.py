"""诊断脚本 — 逐步测试找出问题"""
import sys, os, traceback

PROJECT = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code"
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(PROJECT)

# 1. 检查HSI目录
data_root = os.path.join(PROJECT, "data", "HSIFoodIngr-64")
print(f"[1] data_root: {data_root}")
print(f"    exists: {os.path.exists(data_root)}")
if os.path.exists(data_root):
    print(f"    subdirs: {os.listdir(data_root)}")

# 2. 试着导入
print("\n[2] import hsi_dataset...")
try:
    from data.hsi_dataset import HSIFoodIngrDataset, parse_envi_hdr, read_envi_raw
    print("    OK")
except Exception as e:
    print(f"    FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3. 创建dataset
print("\n[3] create dataset...")
try:
    ds = HSIFoodIngrDataset(data_root, split='train', img_size=256)
    print(f"    train samples: {len(ds)}")
except Exception as e:
    print(f"    FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# 4. 读第一个样本
print("\n[4] read first sample...")
try:
    rgb, nir = ds[0]
    print(f"    rgb: {rgb.shape} [{rgb.min():.2f},{rgb.max():.2f}]")
    print(f"    nir: {nir.shape} [{nir.min():.2f},{nir.max():.2f}]")
except Exception as e:
    print(f"    FAIL: {e}")
    traceback.print_exc()
    
    # 额外诊断：手动解析第一个hdr
    print("\n[4b] 手动解析第一个hdr文件...")
    import glob
    for subdir in sorted(os.listdir(data_root)):
        sub_path = os.path.join(data_root, subdir)
        if not os.path.isdir(sub_path):
            continue
        hdrs = sorted(glob.glob(os.path.join(sub_path, '*.hdr')))
        if hdrs:
            hdr0 = hdrs[0]
            print(f"    hdr: {hdr0}")
            try:
                info = parse_envi_hdr(hdr0)
                print(f"    parsed keys: {list(info.keys())}")
                for k in ['samples', 'lines', 'bands', 'data type', 'interleave']:
                    print(f"    {k}: {info.get(k, 'MISSING')}")
                if 'wavelength' in info:
                    wl = info['wavelength']
                    print(f"    wavelengths: {len(wl)} bands, first={wl[0]}, last={wl[-1]}")
            except Exception as e2:
                print(f"    parse FAIL: {e2}")
                traceback.print_exc()
            break

# 5. 测试模型
print("\n[5] test models...")
try:
    import torch
    from models.generator import UNetGenerator
    from models.discriminator import PatchDiscriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    device: {device}")
    G = UNetGenerator(3, 1).to(device)
    D = PatchDiscriminator(4).to(device)
    print(f"    G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"    D params: {sum(p.numel() for p in D.parameters()):,}")
    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        y = G(x)
    print(f"    G forward: {list(x.shape)} -> {list(y.shape)} ✅")
except Exception as e:
    print(f"    FAIL: {e}")
    traceback.print_exc()

print("\n=== 诊断完成 ===")
