"""Extract Nutrition5k images from data/archive/dish_images.pkl
DataFrame columns: dish (str), rgb_image (bytes JPEG), depth_image (bytes)
Output: data/Nutrition5k/images/{dish_id}_rgb.jpg + {dish_id}_depth.jpg
"""
import os, sys, pickle
import pandas as pd

BASE = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data"
OUT_IMG = os.path.join(BASE, "Nutrition5k", "images")
os.makedirs(OUT_IMG, exist_ok=True)

# --- 1. xlsx → csv (if not already done) ---
for xlsx_name in ["dishes.xlsx", "dish_ingredients.xlsx", "ingredients.xlsx"]:
    src = os.path.join(BASE, "archive", xlsx_name)
    dst = os.path.join(BASE, "Nutrition5k", xlsx_name.replace(".xlsx", ".csv"))
    if os.path.exists(src) and not os.path.exists(dst):
        df = pd.read_excel(src)
        df.to_csv(dst, index=False)
        print(f"[OK] {xlsx_name} → CSV ({len(df)} rows)")
    elif os.path.exists(dst):
        print(f"[SKIP] {xlsx_name.replace('.xlsx','.csv')} already exists")

# --- 2. pkl → jpg ---
pkl_path = os.path.join(BASE, "archive", "dish_images.pkl")
print(f"\nLoading {pkl_path} (2.3GB, ~1 min)...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"DataFrame: {data.shape[0]} dishes, columns: {list(data.columns)}")

count_rgb = 0
count_depth = 0
for idx, row in data.iterrows():
    dish_id = row['dish']
    # Save RGB image
    rgb_path = os.path.join(OUT_IMG, f"{dish_id}_rgb.jpg")
    if not os.path.exists(rgb_path):
        with open(rgb_path, 'wb') as f:
            f.write(row['rgb_image'])
        count_rgb += 1
    # Save depth image
    depth_path = os.path.join(OUT_IMG, f"{dish_id}_depth.jpg")
    if not os.path.exists(depth_path):
        with open(depth_path, 'wb') as f:
            f.write(row['depth_image'])
        count_depth += 1
    if (idx + 1) % 500 == 0:
        print(f"  {idx+1}/{data.shape[0]} processed...")

print(f"\n[DONE] Extracted {count_rgb} RGB + {count_depth} depth images to {OUT_IMG}")

# Verify
total_jpg = len([f for f in os.listdir(OUT_IMG) if f.endswith('.jpg')])
print(f"Verify: {total_jpg} JPGs in {OUT_IMG}")
