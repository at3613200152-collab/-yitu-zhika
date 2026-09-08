"""一图知卡 Phase1 测试脚本 — 测试 DataLoader + Generator + Discriminator"""
import sys
import os

PROJECT = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code"
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(PROJECT)

print("=" * 60)
print("  一图知卡 Phase1 组件测试")
print("=" * 60)

# ── 1. DataLoader ──
print("\n[1/3] DataLoader 测试")
from data.hsi_dataset import HSIFoodIngrDataset
data_root = os.path.join(PROJECT, "data", "HSIFoodIngr-64")
ds = HSIFoodIngrDataset(data_root, split='train', img_size=256)
if len(ds) > 0:
    rgb, nir = ds[0]
    print(f"  ✅ train: {len(ds)} samples")
    print(f"  ✅ rgb: {rgb.shape} range [{rgb.min():.2f}, {rgb.max():.2f}]")
    print(f"  ✅ nir: {nir.shape} range [{nir.min():.2f}, {nir.max():.2f}]")
else:
    print("  ❌ No samples found!")
    sys.exit(1)

# ── 2. Generator ──
print("\n[2/3] Generator 前向传播测试")
import torch
from models.generator import UNetGenerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
G = UNetGenerator(in_channels=3, out_channels=1).to(device)
print(f"  ✅ G params: {sum(p.numel() for p in G.parameters()):,}")
with torch.no_grad():
    x = torch.randn(1, 3, 256, 256, device=device)
    y = G(x)
print(f"  ✅ Input [1,3,256,256] → Output {list(y.shape)} range [{y.min():.3f}, {y.max():.3f}]")

# ── 3. Discriminator ──
print("\n[3/3] Discriminator 前向传播测试")
from models.discriminator import PatchDiscriminator
D = PatchDiscriminator(in_channels=4).to(device)
print(f"  ✅ D params: {sum(p.numel() for p in D.parameters()):,}")
with torch.no_grad():
    rgb_t = torch.randn(1, 3, 256, 256, device=device)
    nir_t = torch.randn(1, 1, 256, 256, device=device)
    out = D(rgb_t, nir_t)
print(f"  ✅ Input [1,3,256,256]+[1,1,256,256] → Output {list(out.shape)}")

# ── 4. 完整前向 + 反向 ──
print("\n[4/4] 完整 G+D 前向反向测试 (1 batch)")
from data.hsi_dataset import build_dataloaders
train_loader, val_loader = build_dataloaders(data_root, batch_size=2, num_workers=0, img_size=256)
real_rgb, real_nir = next(iter(train_loader))
real_rgb, real_nir = real_rgb.to(device), real_nir.to(device)
print(f"  ✅ batch: rgb {list(real_rgb.shape)}, nir {list(real_nir.shape)}")

# G forward
fake_nir = G(real_rgb)
loss_l1 = torch.nn.L1Loss()(fake_nir, real_nir)
print(f"  ✅ G forward: L1={loss_l1.item():.4f}")

# D forward
pred = D(real_rgb, fake_nir)
loss_d = torch.nn.BCEWithLogitsLoss()(pred, torch.ones_like(pred))
print(f"  ✅ D forward: loss={loss_d.item():.4f}")

# backward
loss_g = loss_d + 100 * loss_l1
loss_g.backward()
print(f"  ✅ backward OK")

print("\n" + "=" * 60)
print("  🎉 全部测试通过! 可以开始训练了")
print("=" * 60)
