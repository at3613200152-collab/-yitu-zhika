"""生成带食物特征的训练数据（无需下载数据集即可训练）"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from inference.food_knowledge import FOOD_KNOWLEDGE_BASE

random.seed(42)
np.random.seed(42)

FOOD_COLORS = {
    "apple":        [(200, 50, 60), (180, 40, 50), (210, 60, 70)],
    "banana":       [(240, 220, 80), (230, 210, 70), (250, 230, 90)],
    "orange":       [(255, 165, 0), (240, 150, 10), (255, 175, 20)],
    "strawberry":   [(220, 30, 40), (200, 20, 30), (230, 40, 50)],
    "grape":        [(128, 0, 128), (110, 0, 120), (140, 10, 140)],
    "watermelon":   [(200, 50, 60), (50, 200, 50), (255, 200, 200)],
    "broccoli":     [(60, 160, 60), (50, 140, 50), (70, 180, 70)],
    "cucumber":     [(80, 180, 80), (70, 170, 70), (90, 190, 90)],
    "carrot":       [(230, 120, 30), (210, 100, 20), (240, 130, 40)],
    "tomato":       [(220, 50, 50), (200, 40, 40), (230, 60, 60)],
    "corn":         [(255, 220, 50), (240, 200, 40), (255, 230, 60)],
    "potato":       [(180, 140, 90), (170, 130, 80), (190, 150, 100)],
    "bread":        [(210, 170, 120), (200, 160, 110), (220, 180, 130)],
    "rice":         [(250, 245, 235), (240, 235, 225), (255, 250, 240)],
    "pasta":       [(230, 190, 100), (220, 180, 90), (240, 200, 110)],
    "chicken":      [(230, 190, 140), (220, 180, 130), (240, 200, 150)],
    "beef":         [(180, 80, 60), (160, 70, 50), (190, 90, 70)],
    "pork":         [(220, 150, 160), (210, 140, 150), (230, 160, 170)],
    "steak":        [(160, 70, 50), (140, 60, 40), (170, 80, 60)],
    "fish":         [(180, 180, 200), (170, 170, 190), (190, 190, 210)],
    "shrimp":       [(230, 150, 120), (220, 140, 110), (240, 160, 130)],
    "cheese":       [(255, 200, 80), (240, 190, 70), (255, 210, 90)],
    "milk":         [(250, 250, 250), (240, 240, 240), (255, 255, 255)],
    "yogurt":       [(250, 245, 240), (240, 235, 230), (255, 250, 245)],
    "egg":          [(250, 230, 180), (240, 220, 170), (255, 240, 190)],
    "tofu":         [(245, 240, 230), (235, 230, 220), (250, 245, 235)],
    "salad":        [(100, 180, 60), (120, 200, 80), (80, 160, 40)],
    "soup":         [(180, 120, 60), (160, 100, 50), (200, 140, 70)],
    "hamburger":    [(180, 120, 60), (200, 160, 100), (100, 180, 60)],
    "pizza":        [(220, 160, 80), (200, 60, 50), (240, 200, 100)],
    "sandwich":     [(200, 160, 100), (180, 140, 80), (220, 180, 120)],
    "ice_cream":    [(250, 240, 230), (230, 190, 150), (200, 150, 200)],
    "cake":         [(230, 190, 150), (250, 240, 230), (180, 100, 60)],
    "chocolate":    [(100, 60, 30), (80, 50, 20), (120, 70, 40)],
    "donut":        [(220, 150, 100), (200, 130, 80), (240, 170, 120)],
    "cookie":       [(200, 160, 100), (180, 140, 80), (220, 180, 120)],
    "nuts":         [(180, 140, 90), (160, 120, 70), (200, 160, 110)],
    "sushi":        [(250, 250, 250), (220, 80, 60), (200, 200, 200)],
    "dumpling":     [(240, 230, 210), (230, 220, 200), (250, 240, 220)],
    "noodle":       [(230, 200, 150), (220, 190, 140), (240, 210, 160)],
    "pancake":      [(210, 170, 110), (190, 150, 90), (220, 180, 120)],
    "waffle":       [(220, 180, 120), (200, 160, 100), (230, 190, 130)],
    "croissant":    [(220, 170, 110), (200, 150, 90), (230, 180, 120)],
    "muffin":       [(180, 120, 70), (160, 100, 60), (200, 140, 80)],
    "pie":          [(220, 160, 80), (180, 120, 60), (240, 180, 100)],
    "spinach":      [(50, 140, 50), (40, 120, 40), (60, 160, 60)],
    "mushroom":     [(200, 180, 150), (180, 160, 130), (220, 200, 170)],
    "onion":        [(220, 200, 180), (200, 180, 160), (230, 210, 190)],
    "garlic":       [(230, 220, 200), (210, 200, 180), (240, 230, 210)],
    "avocado":       [(80, 130, 60), (100, 150, 70), (60, 110, 50)],
    "lobster":      [(220, 60, 40), (200, 50, 30), (230, 70, 50)],
    "crab":         [(230, 80, 50), (210, 70, 40), (240, 90, 60)],
    "lamb":         [(180, 100, 70), (160, 90, 60), (190, 110, 80)],
    "cabbage":      [(180, 200, 140), (160, 180, 120), (200, 220, 160)],
    "cauliflower":  [(240, 235, 220), (230, 225, 210), (250, 245, 230)],
    "peas":         [(100, 180, 60), (80, 160, 40), (120, 200, 80)],
    "beans":        [(150, 100, 60), (130, 80, 50), (170, 120, 70)],
    "seeds":        [(160, 120, 70), (140, 100, 60), (180, 140, 80)],
    "popcorn":      [(240, 220, 180), (220, 200, 160), (250, 230, 190)],
    "olive":        [(80, 80, 40), (60, 60, 30), (100, 100, 50)],
    "bread_roll":   [(210, 170, 120), (200, 160, 110), (220, 180, 130)],
    "rice_cake":    [(240, 230, 210), (230, 220, 200), (250, 240, 220)],
    "cheese_cake":  [(250, 240, 220), (230, 200, 150), (240, 210, 160)],
    "strawberry_juice": [(220, 60, 60), (200, 50, 50), (240, 70, 70)],
    "pineapple":    [(220, 180, 50), (200, 160, 40), (240, 200, 60)],
}


def generate_food_image(food_key, size=256):
    """为指定食物生成程序化图片"""
    colors = FOOD_COLORS.get(food_key, [(150, 150, 150), (130, 130, 130), (170, 170, 170)])
    info = FOOD_KNOWLEDGE_BASE.get(food_key, {})
    category = info.get("category", "其他")
    
    img = Image.new("RGB", (size, size), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    cx, cy = size // 2, size // 2
    
    if category == "水果":
        base_color = colors[0]
        draw.ellipse([cx-90, cy-90, cx+90, cy+90], fill=base_color)
        for _ in range(20):
            x = cx + random.randint(-70, 70)
            y = cy + random.randint(-70, 70)
            r = random.randint(10, 25)
            c = random.choice(colors)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=c)
        draw.ellipse([cx-85, cy-85, cx+85, cy+85], outline=colors[1], width=2)
        if food_key in ["apple", "orange", "peach"]:
            draw.rectangle([cx-3, cy-100, cx+3, cy-80], fill=(100, 70, 30))
    
    elif category == "蔬菜":
        base_color = colors[0]
        if food_key in ["broccoli", "cauliflower"]:
            for _ in range(8):
                x = cx + random.randint(-60, 60)
                y = cy + random.randint(-60, 60)
                r = random.randint(30, 50)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
        elif food_key in ["carrot"]:
            draw.polygon([(cx-30, cy+80), (cx+30, cy+80), (cx, cy-80)], fill=base_color)
            for _ in range(10):
                x = cx + random.randint(-20, 20)
                y = cy + random.randint(-60, 60)
                r = random.randint(5, 15)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
            draw.line([cx-5, cy-80, cx-20, cy-110], fill=(80, 160, 50), width=5)
            draw.line([cx+5, cy-80, cx+15, cy-110], fill=(80, 160, 50), width=5)
        elif food_key in ["cucumber", "corn"]:
            draw.ellipse([cx-40, cy-100, cx+40, cy+100], fill=base_color)
            for _ in range(15):
                x = cx + random.randint(-30, 30)
                y = cy + random.randint(-80, 80)
                r = random.randint(5, 15)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
        else:
            draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=base_color)
            for _ in range(15):
                x = cx + random.randint(-60, 60)
                y = cy + random.randint(-60, 60)
                r = random.randint(10, 25)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    elif category == "肉类":
        base_color = colors[0]
        draw.ellipse([cx-90, cy-70, cx+90, cy+70], fill=base_color)
        for _ in range(25):
            x = cx + random.randint(-70, 70)
            y = cy + random.randint(-50, 50)
            r = random.randint(8, 20)
            c = random.choice(colors)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=c)
        if food_key == "steak":
            draw.line([cx-60, cy-30, cx+60, cy-30], fill=colors[2], width=3)
            draw.line([cx-60, cy, cx+60, cy], fill=colors[2], width=3)
            draw.line([cx-60, cy+30, cx+60, cy+30], fill=colors[2], width=3)
    
    elif category == "海鲜":
        base_color = colors[0]
        if food_key in ["fish"]:
            draw.ellipse([cx-90, cy-50, cx+90, cy+50], fill=base_color)
            draw.polygon([(cx+90, cy), (cx+120, cy-30), (cx+120, cy+30)], fill=base_color)
            for _ in range(15):
                x = cx + random.randint(-70, 70)
                y = cy + random.randint(-30, 30)
                r = random.randint(5, 12)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
        else:
            for _ in range(8):
                angle = random.uniform(0, 6.28)
                r = random.randint(30, 60)
                x = cx + int(r * np.cos(angle))
                y = cy + int(r * np.sin(angle))
                size_r = random.randint(20, 35)
                draw.ellipse([x-size_r, y-size_r, x+size_r, y+size_r], fill=random.choice(colors))
    
    elif category == "主食":
        base_color = colors[0]
        if food_key in ["rice", "rice_cake"]:
            for _ in range(60):
                x = random.randint(20, size-20)
                y = random.randint(20, size-20)
                r = random.randint(4, 8)
                c = random.choice(colors)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=c)
            draw.ellipse([cx-100, cy-100, cx+100, cy+100], outline=(200, 180, 140), width=3)
        elif food_key in ["bread", "bread_roll", "croissant"]:
            draw.ellipse([cx-90, cy-60, cx+90, cy+60], fill=base_color)
            for _ in range(15):
                x = cx + random.randint(-70, 70)
                y = cy + random.randint(-40, 40)
                r = random.randint(10, 25)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
        else:
            draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=base_color)
            for _ in range(20):
                x = cx + random.randint(-60, 60)
                y = cy + random.randint(-60, 60)
                r = random.randint(8, 20)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    elif category == "快餐":
        base_color = colors[0]
        if food_key == "pizza":
            draw.ellipse([cx-100, cy-100, cx+100, cy+100], fill=base_color)
            for _ in range(15):
                x = cx + random.randint(-70, 70)
                y = cy + random.randint(-70, 70)
                r = random.randint(8, 15)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=colors[1])
            for _ in range(10):
                x = cx + random.randint(-60, 60)
                y = cy + random.randint(-60, 60)
                r = random.randint(5, 10)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(100, 70, 30))
        elif food_key == "hamburger":
            draw.ellipse([cx-100, cy-40, cx+100, cy+40], fill=(200, 160, 100))
            draw.ellipse([cx-100, cy-60, cx+100, cy-20], fill=(100, 180, 60))
            draw.ellipse([cx-100, cy-80, cx+100, cy-40], fill=base_color)
            draw.ellipse([cx-100, cy+20, cx+100, cy+60], fill=(180, 100, 60))
            draw.ellipse([cx-100, cy+60, cx+100, cy+100], fill=(200, 160, 100))
        else:
            draw.ellipse([cx-90, cy-90, cx+90, cy+90], fill=base_color)
            for _ in range(15):
                x = cx + random.randint(-70, 70)
                y = cy + random.randint(-70, 70)
                r = random.randint(8, 20)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    elif category == "甜品":
        base_color = colors[0]
        if food_key in ["ice_cream"]:
            draw.ellipse([cx-60, cy-40, cx+60, cy+40], fill=base_color)
            draw.polygon([(cx-50, cy+20), (cx+50, cy+20), (cx, cy+120)], fill=(180, 130, 90))
        elif food_key in ["cake", "cheese_cake", "pie"]:
            draw.rectangle([cx-80, cy-60, cx+80, cy+60], fill=base_color)
            draw.rectangle([cx-80, cy-70, cx+80, cy-60], fill=colors[1])
            for _ in range(10):
                x = cx + random.randint(-60, 60)
                y = cy + random.randint(-40, 40)
                r = random.randint(5, 12)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
        elif food_key in ["donut"]:
            draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=base_color)
            draw.ellipse([cx-25, cy-25, cx+25, cy+25], fill=(30, 30, 30))
        else:
            draw.ellipse([cx-70, cy-70, cx+70, cy+70], fill=base_color)
            for _ in range(10):
                x = cx + random.randint(-50, 50)
                y = cy + random.randint(-50, 50)
                r = random.randint(8, 18)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    elif category == "乳制品":
        base_color = colors[0]
        draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=base_color)
        for _ in range(15):
            x = cx + random.randint(-60, 60)
            y = cy + random.randint(-60, 60)
            r = random.randint(8, 20)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    elif category == "蛋类":
        base_color = colors[0]
        draw.ellipse([cx-80, cy-60, cx+80, cy+60], fill=(250, 245, 230))
        draw.ellipse([cx-40, cy-40, cx+40, cy+40], fill=base_color)
    
    elif category == "豆制品":
        base_color = colors[0]
        for _ in range(6):
            x = cx + random.randint(-50, 50)
            y = cy + random.randint(-50, 50)
            r = random.randint(30, 45)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    elif category == "汤品":
        base_color = colors[0]
        draw.ellipse([cx-100, cy-80, cx+100, cy+80], fill=base_color)
        for _ in range(20):
            x = cx + random.randint(-80, 80)
            y = cy + random.randint(-60, 60)
            r = random.randint(5, 15)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    else:
        base_color = colors[0]
        draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=base_color)
        for _ in range(15):
            x = cx + random.randint(-60, 60)
            y = cy + random.randint(-60, 60)
            r = random.randint(8, 20)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    
    noise = np.random.normal(0, 10, (size, size, 3))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=(30, 30, 30))
    
    return img


def generate_nir_from_rgb(rgb_img):
    """从RGB图像生成模拟NIR图像"""
    rgb_array = np.array(rgb_img, dtype=np.float32)
    
    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    
    nir = 0.3 * r + 0.5 * g + 0.2 * b
    
    nir = nir + np.random.normal(0, 5, nir.shape)
    
    contrast_factor = np.random.uniform(0.8, 1.3)
    nir = (nir - 128) * contrast_factor + 128
    
    nir = np.clip(nir, 0, 255).astype(np.uint8)
    
    nir_img = Image.fromarray(nir)
    
    if random.random() > 0.5:
        nir_img = nir_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
    
    return nir_img


def generate_hsi_dataset(num_per_class=20):
    """生成 HSIFoodIngr-64 数据集"""
    rgb_dir = "data/HSIFoodIngr-64/RGB"
    nir_dir = "data/HSIFoodIngr-64/NIR"
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(nir_dir, exist_ok=True)
    
    food_keys = list(FOOD_KNOWLEDGE_BASE.keys())
    print(f"Generating HSI dataset: {len(food_keys)} foods x {num_per_class} images = {len(food_keys) * num_per_class} total")
    
    count = 0
    for food_key in food_keys:
        for i in range(num_per_class):
            rgb_img = generate_food_image(food_key, size=256)
            nir_img = generate_nir_from_rgb(rgb_img)
            
            fname = f"{food_key}_{i:04d}.png"
            rgb_img.save(os.path.join(rgb_dir, fname))
            nir_img.save(os.path.join(nir_dir, fname))
            count += 1
        
        if count % 100 == 0:
            print(f"  Generated {count} images...")
    
    print(f"HSI dataset generated: {count} images in {rgb_dir} and {nir_dir}")


def generate_nutrition5k_dataset(num_per_class=30):
    """生成 Nutrition5k 数据集"""
    rgb_dir = "data/Nutrition5k/images"
    labels_file = "data/Nutrition5k/labels.csv"
    
    os.makedirs(rgb_dir, exist_ok=True)
    
    food_keys = list(FOOD_KNOWLEDGE_BASE.keys())
    print(f"Generating Nutrition5k dataset: {len(food_keys)} foods x {num_per_class} images = {len(food_keys) * num_per_class} total")
    
    import pandas as pd
    
    data = []
    count = 0
    for cls_idx, food_key in enumerate(food_keys):
        info = FOOD_KNOWLEDGE_BASE[food_key]
        base_cal = info["typical_calories"]
        base_weight = info["typical_weight"]
        
        for i in range(num_per_class):
            rgb_img = generate_food_image(food_key, size=224)
            
            cal_noise = np.random.uniform(0.7, 1.3)
            weight_noise = np.random.uniform(0.7, 1.3)
            
            calories = round(base_cal * cal_noise, 1)
            weight = round(base_weight * weight_noise, 1)
            
            fname = f"{food_key}_{i:04d}.png"
            rgb_img.save(os.path.join(rgb_dir, fname))
            
            data.append({
                "image_path": fname,
                "class": cls_idx,
                "class_name": food_key,
                "calories": calories,
                "weight": weight,
            })
            count += 1
        
        if count % 100 == 0:
            print(f"  Generated {count} images...")
    
    df = pd.DataFrame(data)
    df.to_csv(labels_file, index=False)
    print(f"Nutrition5k dataset generated: {count} images, {len(df)} labels")


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Training Data")
    print("=" * 60)
    
    print("\n[1/2] Generating HSI dataset (RGB + NIR pairs)...")
    generate_hsi_dataset(num_per_class=15)
    
    print("\n[2/2] Generating Nutrition5k dataset (food images + labels)...")
    generate_nutrition5k_dataset(num_per_class=25)
    
    print("\n" + "=" * 60)
    print("  Data generation completed!")
    print("=" * 60)
