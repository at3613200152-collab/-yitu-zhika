"""Inspect dish_images.pkl structure and find HSI tar.gz files."""
import os, pickle, pandas as pd

# --- 1. Inspect pkl ---
pkl_path = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data\archive\dish_images.pkl"
print("Loading pkl...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

if isinstance(data, pd.DataFrame):
    print(f"\nColumns: {list(data.columns)}")
    print(f"Dtypes:\n{data.dtypes}")
    print(f"\nFirst row sample:")
    for col in data.columns:
        val = data.iloc[0][col]
        if isinstance(val, bytes):
            print(f"  {col}: bytes len={len(val)}")
        elif hasattr(val, 'shape'):
            print(f"  {col}: ndarray shape={val.shape} dtype={val.dtype}")
        else:
            print(f"  {col}: {type(val).__name__} = {str(val)[:100]}")
    # Check if any column has image bytes
    for col in data.columns:
        sample = data.iloc[0][col]
        if isinstance(sample, bytes) and len(sample) > 1000:
            print(f"\n  *** Column '{col}' looks like image bytes! (first row: {len(sample)} bytes)")
        if hasattr(sample, 'shape') and len(sample.shape) >= 2:
            print(f"\n  *** Column '{col}' looks like image array! shape={sample.shape}")

# --- 2. Find HSI tar.gz ---
data_base = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data"
hsi_dir = os.path.join(data_base, "HSIFoodIngr-64")
print(f"\n=== HSI directory contents ===")
if os.path.isdir(hsi_dir):
    for item in os.listdir(hsi_dir):
        fp = os.path.join(hsi_dir, item)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            print(f"  FILE: {item}  ({sz/1048576:.0f}MB)")
        elif os.path.isdir(fp):
            sub = os.listdir(fp)
            print(f"  DIR:  {item}/ ({len(sub)} items)")
            for s in sub[:10]:
                sp = os.path.join(fp, s)
                ssz = os.path.getsize(sp) if os.path.isfile(sp) else 0
                tag = f"  ({ssz/1048576:.0f}MB)" if ssz > 1048576 else ""
                print(f"    {s}{tag}")
            if len(sub) > 10:
                print(f"    ... and {len(sub)-10} more")

# Also check data root for any tar.gz
print(f"\n=== tar.gz in data/ root ===")
for f in os.listdir(data_base):
    if f.endswith('.tar.gz') or f.endswith('.zip'):
        sz = os.path.getsize(os.path.join(data_base, f))
        print(f"  {f}: {sz/1048576:.0f}MB")
