"""Inspect the structure of real Nutrition5k data files."""
import pickle
import pandas as pd
import numpy as np

print("=== dish_images.pkl ===")
with open('data/archive/dish_images.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
if isinstance(data, dict):
    keys = list(data.keys())
    print(f"Num keys: {len(keys)}")
    print(f"First 5 keys: {keys[:5]}")
    for k in keys[:3]:
        v = data[k]
        t = type(v).__name__
        if hasattr(v, 'shape'):
            print(f"  {k}: type={t}, shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}")
        elif hasattr(v, '__len__'):
            print(f"  {k}: type={t}, len={len(v)}")
        else:
            print(f"  {k}: type={t}, value={v}")
elif isinstance(data, list):
    print(f"Length: {len(data)}")
    item = data[0]
    print(f"First item type: {type(item).__name__}")
    if isinstance(item, dict):
        print(f"First item keys: {list(item.keys())}")
    elif isinstance(item, np.ndarray):
        print(f"First item shape: {item.shape}, dtype: {item.dtype}")

print("\n=== dishes.xlsx ===")
df_dishes = pd.read_excel('data/archive/dishes.xlsx')
print(f"Shape: {df_dishes.shape}")
print(f"Columns: {list(df_dishes.columns)}")
print(df_dishes.head(5).to_string())
print(f"\nNum unique dish_ids: {df_dishes.iloc[:, 0].nunique()}")

print("\n=== dish_ingredients.xlsx ===")
df_ingr = pd.read_excel('data/archive/dish_ingredients.xlsx')
print(f"Shape: {df_ingr.shape}")
print(f"Columns: {list(df_ingr.columns)}")
print(df_ingr.head(5).to_string())

print("\n=== ingredients.xlsx ===")
df_i = pd.read_excel('data/archive/ingredients.xlsx')
print(f"Shape: {df_i.shape}")
print(f"Columns: {list(df_i.columns)}")
print(df_i.head(5).to_string())
