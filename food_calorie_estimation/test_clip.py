import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from PIL import Image
import numpy as np

print('Creating test image (apple-like)...')
img = Image.new('RGB', (224, 224), (200, 50, 60))
img.save('test_apple.png')

print('Loading estimator (this may download CLIP model on first run)...')
from inference.estimator import FoodCalorieEstimator
est = FoodCalorieEstimator(config_path='config.yaml')

print()
print('Testing classification...')
top5 = est.classify_food_clip('test_apple.png')
for i, c in enumerate(top5):
    print(f'  {i+1}. {c["name"]} ({c["name_en"]}) - {c["probability"]*100:.1f}%')

print()
print('Testing full prediction...')
result = est.predict_all('test_apple.png')
ms = result["multispectral"]
bl = result["baseline"]
print(f'  Multispectral: {ms["food_class"]} - {ms["calories"]} kcal ({ms["class_probability"]*100:.1f}%)')
print(f'  Baseline:     {bl["food_class"]} - {bl["calories"]} kcal ({bl["class_probability"]*100:.1f}%)')
print(f'  Diff: {result["comparison"]["calories_diff"]} kcal ({result["comparison"]["calories_improvement"]:.1f}%)')
print()
print('All tests passed!')
