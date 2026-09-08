"""Extract Nutrition5k from data/archive/ to data/Nutrition5k/
- dish_images.pkl → data/Nutrition5k/images/dish_{id}.jpg
- dishes.xlsx → data/Nutrition5k/dishes.csv
- dish_ingredients.xlsx → data/Nutrition5k/dish_ingredients.csv
- ingredients.xlsx → data/Nutrition5k/ingredients.csv
"""
import os, pickle, struct, sys
import pandas as pd
from PIL import Image
from io import BytesIO

BASE = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data"
ARCHIVE = os.path.join(BASE, "archive")
OUT = os.path.join(BASE, "Nutrition5k")

# --- 1. Extract xlsx → csv ---
for xlsx_name in ["dishes.xlsx", "dish_ingredients.xlsx", "ingredients.xlsx"]:
    src = os.path.join(ARCHIVE, xlsx_name)
    dst = os.path.join(OUT, xlsx_name.replace(".xlsx", ".csv"))
    if os.path.exists(src):
        df = pd.read_excel(src)
        df.to_csv(dst, index=False)
        print(f"[OK] {xlsx_name} → {len(df)} rows → CSV")
    else:
        print(f"[SKIP] {xlsx_name} not found")

# --- 2. Extract dish_images.pkl → jpg ---
pkl_path = os.path.join(ARCHIVE, "dish_images.pkl")
img_dir = os.path.join(OUT, "images")
os.makedirs(img_dir, exist_ok=True)

if not os.path.exists(pkl_path):
    print(f"[ERROR] {pkl_path} not found!")
    sys.exit(1)

print(f"Loading {pkl_path} (2.3GB, this takes a minute)...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"Type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# Nutrition5k pkl format: dict {dish_id: bytes_jpeg} or list of tuples
count = 0
if isinstance(data, dict):
    for dish_id, img_bytes in data.items():
        if isinstance(img_bytes, bytes):
            out_path = os.path.join(img_dir, f"dish_{dish_id}.jpg")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            count += 1
            if count % 500 == 0:
                print(f"  extracted {count} images...")
elif isinstance(data, list):
    for i, item in enumerate(data):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            dish_id, img_bytes = item[0], item[1]
            if isinstance(img_bytes, bytes):
                out_path = os.path.join(img_dir, f"dish_{dish_id}.jpg")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                count += 1
                if count % 500 == 0:
                    print(f"  extracted {count} images...")
        elif isinstance(item, bytes):
            out_path = os.path.join(img_dir, f"dish_{i}.jpg")
            with open(out_path, "wb") as f:
                f.write(item)
            count += 1

print(f"\n[DONE] Extracted {count} images to {img_dir}")
print(f"CSV files in {OUT}")

# Quick verify
csvs = [f for f in os.listdir(OUT) if f.endswith(".csv")]
jpgs = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) if os.path.isdir(img_dir) else 0
print(f"\nVerify: {len(csvs)} CSVs, {jpgs} JPGs")
