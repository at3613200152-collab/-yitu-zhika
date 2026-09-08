"""
从 dish_images.pkl + dishes.xlsx + dish_ingredients.xlsx 提取真实 Nutrition5k 数据。
- 提取 RGB 图像 → data/Nutrition5k_real/images/
- 合并菜品卡路里/重量 → labels.csv
- 用主要食材名作为食物类别
"""
import os
import pickle
import io
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === 1. 加载真实数据 ===
print("[1/4] 加载 dish_images.pkl ...")
with open('data/archive/dish_images.pkl', 'rb') as f:
    img_df = pickle.load(f)
print(f"  图像 DataFrame: {len(img_df)} 行, 列: {list(img_df.columns)}")

print("[2/4] 加载 dishes.xlsx 和 dish_ingredients.xlsx ...")
dishes_df = pd.read_excel('data/archive/dishes.xlsx')
ingr_df = pd.read_excel('data/archive/dish_ingredients.xlsx')
print(f"  菜品: {len(dishes_df)}, 食材记录: {len(ingr_df)}")

# === 2. 为每道菜确定主要食材（类别）===
print("[3/4] 推导食物类别 ...")
# 每道菜取克数最大的食材作为类别
ingr_sorted = ingr_df.sort_values(['dish_id', 'grams'], ascending=[True, False])
primary_ingr = ingr_sorted.groupby('dish_id').first()['ingr_name'].reset_index()
primary_ingr.columns = ['dish_id', 'category']

# 合并图像 + 营养信息 + 类别
dishes_merged = dishes_df.merge(primary_ingr, on='dish_id', how='left')
dishes_merged['category'] = dishes_merged['category'].fillna('unknown')

# 统计类别分布
cat_counts = dishes_merged['category'].value_counts()
print(f"  原始类别数: {len(cat_counts)}")
print(f"  前 10 类: {dict(cat_counts.head(10))}")

# 保留样本数 >= 10 的类别，其余归为 "other"
valid_cats = cat_counts[cat_counts >= 10].index.tolist()
dishes_merged['category'] = dishes_merged['category'].where(
    dishes_merged['category'].isin(valid_cats), 'other'
)

# 重新编码类别 ID
cat_list = sorted(dishes_merged['category'].unique())
cat2id = {c: i for i, c in enumerate(cat_list)}
dishes_merged['class'] = dishes_merged['category'].map(cat2id)
print(f"  最终类别数: {len(cat_list)}")

# === 3. 提取图像并保存 ===
print("[4/4] 提取真实图像 → data/Nutrition5k_real/images/ ...")
out_dir = 'data/Nutrition5k_real'
img_dir = os.path.join(out_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

# 合并图像数据
img_df = img_df.rename(columns={'dish': 'dish_id'})
merged = img_df.merge(dishes_merged, on='dish_id', how='inner')
print(f"  匹配到的菜品: {len(merged)}")

labels = []
saved = 0
for idx, row in merged.iterrows():
    dish_id = row['dish_id']
    img_bytes = row['rgb_image']

    # 解码图像
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        continue

    # Resize 到 224x224
    img = img.resize((224, 224), Image.LANCZOS)

    fname = f"{dish_id}.png"
    img.save(os.path.join(img_dir, fname))

    labels.append({
        'image_path': fname,
        'class': int(row['class']),
        'class_name': row['category'],
        'calories': float(row['total_calories']),
        'weight': float(row['total_mass']),
    })
    saved += 1

    if saved % 500 == 0:
        print(f"  已保存 {saved} / {len(merged)} 张图像")

print(f"\n=== 提取完成: {saved} 张真实食物图像 ===")

# === 4. 保存标签 CSV ===
labels_df = pd.DataFrame(labels)
labels_csv = os.path.join(out_dir, 'labels.csv')
labels_df.to_csv(labels_csv, index=False)
print(f"标签文件: {labels_csv}")

# 统计
print(f"\n=== 数据集统计 ===")
print(f"总图像数: {len(labels_df)}")
print(f"类别数: {labels_df['class'].nunique()}")
print(f"卡路里范围: {labels_df['calories'].min():.0f} ~ {labels_df['calories'].max():.0f}")
print(f"重量范围: {labels_df['weight'].min():.0f} ~ {labels_df['weight'].max():.0f}g")
print(f"\n每类样本数:")
for cn, cnt in labels_df.groupby('class_name').size().sort_values(ascending=False).head(15).items():
    print(f"  {cn:20s}: {cnt:4d}")
