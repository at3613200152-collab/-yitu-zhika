"""画训练曲线图"""
import os
import sys

LOG_DIR = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\logs\phase1"
OUT_DIR = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\training_curves.png"

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard未安装，无法读取日志")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ea = EventAccumulator(LOG_DIR)
ea.Reload()

tags = [t for t in ea.Tags()['scalars']]
print(f"Available tags: {tags}")

# 收集数据
curves = {}
for tag in tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    curves[tag] = (steps, values)

# 画图
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Phase1 Training Curves (RGB→NIR Generator)', fontsize=14, fontweight='bold')

plot_configs = [
    ('epoch/G_loss', 'G Loss (Generator)', 0, 0),
    ('epoch/D_loss', 'D Loss (Discriminator)', 0, 1),
    ('val/L1', 'Val L1 (lower=better)', 0, 2),
    ('val/PSNR', 'Val PSNR dB (higher=better)', 1, 0),
    ('train/G_L1', 'Train L1 Loss', 1, 1),
    ('train/G_GAN', 'Train GAN Loss', 1, 2),
]

for tag, title, row, col in plot_configs:
    ax = axes[row][col]
    if tag in curves:
        steps, values = curves[tag]
        ax.plot(steps, values, linewidth=1.5, color='steelblue')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR, dpi=150, bbox_inches='tight')
print(f"\n✅ 图已保存: {OUT_DIR}")
