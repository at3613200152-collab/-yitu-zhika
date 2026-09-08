"""Remap class IDs to be sequential 0..N-1."""
import pandas as pd

df = pd.read_csv('data/Nutrition5k_real/labels.csv')
cats = sorted(df['class'].unique())
cat2new = {c: i for i, c in enumerate(cats)}
df['class'] = df['class'].map(cat2new)
df.to_csv('data/Nutrition5k_real/labels.csv', index=False)
print(f"Remapped: {df['class'].nunique()} classes, range: {df['class'].min()}-{df['class'].max()}")
