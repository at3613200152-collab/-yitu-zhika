"""勘察：为 label_schema 增加 fruit 类做证据准备。

- 列出含水果/果汁/沙拉词的食材，及其当前分类、总克数、出现餐盘数
- 只读 data/Nutrition5k/dish_ingredients.csv，不写任何产物
用法：python scripts/probe_fruit_scope.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.build_categories import classify_ingredient, match_ingredient

KEYS = ['melon', 'apple', 'berry', 'grape', 'orange', 'banana', 'fruit', 'juice',
        'peach', 'pear', 'mango', 'kiwi', 'cherry', 'straw', 'pine', 'lemon', 'lime',
        'avocado', 'raisin', 'cranberr', 'blueberr', 'raspberr', 'nectarine', 'plum',
        'apricot', 'coconut', 'fig', 'date ', 'salad', 'tomato', 'olive', 'corn',
        'potato', 'bean', 'pea ', 'pepper', 'cucumber', 'lettuce', 'spinach', 'onion',
        'carrot', 'broccoli', 'cauliflower', 'mushroom', 'zucchini', 'squash', 'pumpkin']


def main():
    rows = list(csv.DictReader((ROOT / 'data/Nutrition5k/dish_ingredients.csv').open(encoding='utf-8')))
    grams, dishes = defaultdict(float), defaultdict(set)
    for r in rows:
        name = r['ingr_name'].strip().lower()
        grams[name] += float(r['grams'] or 0)
        dishes[name].add(r['dish_id'])
    hits = [n for n in grams if any(k in n for k in KEYS)]
    print(f'总食材(去重)={len(grams)}  匹配到勘察词的食材={len(hits)}')
    print(f"{'ingredient':32s} {'current':16s} {'match':22s} {'dishes':>6s} {'kg':>8s}")
    for n in sorted(hits, key=lambda x: -grams[x]):
        mm = match_ingredient(n)
        print(f'{n:32s} {classify_ingredient(n):16s} {str(mm):22s} {len(dishes[n]):6d} {grams[n]/1000:8.2f}')
    print('\n当前分类分布（食材数）:')
    dist = defaultdict(int)
    for n in grams:
        dist[classify_ingredient(n)] += 1
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f'  {k:16s} {v}')


if __name__ == '__main__':
    main()
