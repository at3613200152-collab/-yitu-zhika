"""Inspect dish_images.pkl DataFrame structure."""
import os, pickle, pandas as pd

pkl_path = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data\archive\dish_images.pkl"
print("Loading pkl (2.3GB, ~1 min)...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
if hasattr(data, 'shape'):
    print(f"Shape: {data.shape}")
if isinstance(data, pd.DataFrame):
    print(f"Columns: {list(data.columns)}")
    print(f"\nDtypes:")
    for col in data.columns:
        print(f"  {col}: {data[col].dtype}")
    print(f"\nFirst 3 rows sample:")
    for idx in range(min(3, len(data))):
        print(f"\n--- Row {idx} ---")
        for col in data.columns:
            val = data.iloc[idx][col]
            if isinstance(val, bytes):
                print(f"  {col}: bytes len={len(val)}")
            elif hasattr(val, 'shape'):
                print(f"  {col}: ndarray shape={val.shape} dtype={val.dtype}")
            else:
                s = str(val)
                print(f"  {col}: {type(val).__name__} = {s[:120]}")
elif isinstance(data, dict):
    keys = list(data.keys())[:5]
    print(f"Keys (first 5): {keys}")
    for k in keys:
        v = data[k]
        print(f"  {k}: {type(v).__name__}, len/shape={len(v) if hasattr(v,'__len__') else 'N/A'}")
