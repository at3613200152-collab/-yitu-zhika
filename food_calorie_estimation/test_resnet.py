import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from PIL import Image
import numpy as np

print('=' * 60)
print('  Testing Food Calorie Estimator')
print('=' * 60)

print('\n[1/4] Creating test images...')
img_apple = Image.new('RGB', (224, 224), (200, 50, 60))
img_apple.save('test_apple.png')

img_banana = Image.new('RGB', (224, 224), (240, 220, 80))
img_banana.save('test_banana.png')

img_green = Image.new('RGB', (224, 224), (80, 180, 80))
img_green.save('test_veg.png')

print('  Created: test_apple.png, test_banana.png, test_veg.png')

print('\n[2/4] Loading estimator...')
from inference.estimator import FoodCalorieEstimator
est = FoodCalorieEstimator(config_path='config.yaml')

print('\n[3/4] Testing food classification...')
test_images = [
    ('苹果 (红色)', 'test_apple.png'),
    ('香蕉 (黄色)', 'test_banana.png'),
    ('绿色蔬菜', 'test_veg.png'),
]

for name, path in test_images:
    print(f'\n  Test: {name}')
    top5 = est.classify_food(path)
    for i, c in enumerate(top5[:3]):
        print(f'    {i+1}. {c["name"]} ({c["name_en"]}) - {c["probability"]*100:.1f}%')

print('\n[4/4] Testing full prediction...')
result = est.predict_all('test_apple.png')
ms = result["multispectral"]
bl = result["baseline"]
comp = result["comparison"]

print(f'\n  Multispectral (多光谱):')
print(f'    食物: {ms["food_class"]} ({ms["food_class_en"]})')
print(f'    置信度: {ms["class_probability"]*100:.1f}%')
print(f'    卡路里: {ms["calories"]} kcal')
print(f'    重量: {ms["weight"]} g')

print(f'\n  Baseline (基线):')
print(f'    食物: {bl["food_class"]} ({bl["food_class_en"]})')
print(f'    置信度: {bl["class_probability"]*100:.1f}%')
print(f'    卡路里: {bl["calories"]} kcal')
print(f'    重量: {bl["weight"]} g')

print(f'\n  对比:')
print(f'    卡路里差异: {comp["calories_diff"]} kcal')
print(f'    相对变化: {comp["calories_improvement"]:.1f}%')
print(f'    更优方法: {comp["method_better"]}')

print('\n' + '=' * 60)
print('  All tests passed!')
print('=' * 60)
