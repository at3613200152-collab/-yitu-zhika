"""Inspect dish_images.pkl structure."""
import pickle
import pandas as pd

with open('data/archive/dish_images.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Type: {type(df)}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nHead:\n{df.head(3).to_string()}")

# Check first row
print("\n=== First row details ===")
for c in df.columns:
    val = df.iloc[0][c]
    t = type(val).__name__
    shape = getattr(val, 'shape', 'N/A')
    print(f"  {c}: type={t}, shape={shape}")

# Check if there are image columns
for c in df.columns:
    sample = df.iloc[0][c]
    if hasattr(sample, 'shape') and len(sample.shape) >= 2:
        print(f"\n  Image column '{c}': shape={sample.shape}, dtype={sample.dtype}")
    elif isinstance(sample, bytes):
        print(f"\n  Bytes column '{c}': len={len(sample)}")
