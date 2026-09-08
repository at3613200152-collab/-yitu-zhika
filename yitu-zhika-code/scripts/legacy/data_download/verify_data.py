"""Verify data + PyTorch (fixed for Blackwell/sm_120)."""
import os, sys

BASE = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data"

def check_hsi():
    print("\n=== HSIFoodIngr-64 ===")
    hsi_dir = os.path.join(BASE, "HSIFoodIngr-64")
    if not os.path.isdir(hsi_dir):
        print("[!] Directory not found"); return False
    subdirs = [d for d in os.listdir(hsi_dir) if os.path.isdir(os.path.join(hsi_dir, d))]
    print(f"  Subdirectories: {len(subdirs)}")
    for d in sorted(subdirs):
        print(f"    {d}/")
    # Check for .hdr/.raw
    hdr_count = 0
    for dp, dn, fn in os.walk(hsi_dir):
        hdr_count += sum(1 for f in fn if f.endswith('.hdr'))
    print(f"  .hdr files: {hdr_count}")
    print(f"  [OK]" if hdr_count > 0 else "  [!] No .hdr files found")
    return hdr_count > 0

def check_n5k():
    print("\n=== Nutrition5k ===")
    n5k_dir = os.path.join(BASE, "Nutrition5k")
    if not os.path.isdir(n5k_dir):
        print("[!] Directory not found"); return False
    items = os.listdir(n5k_dir)
    print(f"  Contents: {items}")
    # Check CSVs
    csvs = [f for f in items if f.endswith('.csv')]
    print(f"  CSVs: {len(csvs)}")
    # Check images
    img_dir = os.path.join(n5k_dir, "images")
    jpg_count = 0
    if os.path.isdir(img_dir):
        jpg_count = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    print(f"  Images: {jpg_count} JPGs")
    ok = len(csvs) >= 3 and jpg_count > 0
    print(f"  [OK]" if ok else f"  [!] CSVs={len(csvs)}, Images={jpg_count}")
    return ok

def check_pytorch():
    print("\n=== PyTorch & GPU ===")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            try:
                props = torch.cuda.get_device_properties(0)
                vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
                print(f"  VRAM: {vram // 1048576}MB")
            except Exception as e:
                print(f"  VRAM: (could not read: {e})")
        # Quick tensor test
        try:
            x = torch.randn(100, 100, device='cuda')
            y = x @ x.T
            print(f"  [OK] GPU tensor compute works")
            return True
        except Exception as e:
            print(f"  [!] GPU compute failed: {e}")
            print(f"  → Need PyTorch nightly for RTX 5060 (Blackwell sm_120)")
            print(f"  → pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
            return False
    except ImportError:
        print("  [!] PyTorch not installed"); return False

if __name__ == "__main__":
    hsi_ok = check_hsi()
    n5k_ok = check_n5k()
    pt_ok = check_pytorch()
    print(f"\n{'='*40}")
    print(f"HSI: {'✅' if hsi_ok else '❌'}  N5K: {'✅' if n5k_ok else '❌'}  PyTorch: {'✅' if pt_ok else '❌'}")
