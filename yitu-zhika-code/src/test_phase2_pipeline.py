"""Phase2 pipeline quick validation - test full chain in one shot"""
import os, sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

import torch

print(f"PyTorch {torch.__version__}")
cuda_ok = torch.cuda.is_available()
print(f"CUDA: {cuda_ok}" + (f" - {torch.cuda.get_device_name(0)}" if cuda_ok else ""))
device = torch.device('cuda' if cuda_ok else 'cpu')

# ========== 1. DataLoader ==========
print("\n=== 1. DataLoader ===")
from data.nutrition5k_loader import create_nutrition5k_dataloader
train_loader = create_nutrition5k_dataloader(
    root_dir='./data/Nutrition5k', split='train', img_size=256,
    batch_size=4, num_workers=0, augmentation=True,
)
print(f"Train samples: {len(train_loader.dataset)}")
batch = next(iter(train_loader))
images = batch['image']
print(f"  image: {images.shape}, dtype={images.dtype}")
print(f"  calories: {batch['calories'].tolist()}")
print(f"  mass: {batch['mass'].tolist()}")

# ========== 2. Generator ==========
print("\n=== 2. Load Generator ===")
from models.generator import UNetGenerator
gen = UNetGenerator(in_channels=3, out_channels=1, base_filters=64).to(device)
ckpt = torch.load('checkpoints/phase1/final_model.pth', map_location=device)
gen_state = ckpt.get('G_state_dict', ckpt.get('model_state_dict', ckpt))
gen.load_state_dict(gen_state, strict=False)
gen.eval()
n = sum(p.numel() for p in gen.parameters())
print(f"Generator: {n:,} params, epoch={ckpt.get('epoch', '?')}")

# ========== 3. Forward + 4ch ==========
print("\n=== 3. Generator Forward + 4ch ===")
images = images.to(device)
mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
rgb_01 = images * std_t + mean_t
rgb_norm = rgb_01 * 2 - 1
with torch.no_grad():
    nir = gen(rgb_norm)
print(f"  NIR: {nir.shape}, range=[{nir.min():.4f}, {nir.max():.4f}]")
nir_01 = (nir + 1) / 2
nir_imagenet = (nir_01 - 0.485) / 0.229
input_4ch = torch.cat([images, nir_imagenet], dim=1)
print(f"  4ch: {input_4ch.shape}")
print(f"  RGB range=[{images.min():.3f}, {images.max():.3f}], NIR range=[{nir_imagenet.min():.3f}, {nir_imagenet.max():.3f}]")

# ========== 4. ResNetMultiTask ==========
print("\n=== 4. ResNetMultiTask ===")
from models.resnet_multitask import ResNetMultiTask
model = ResNetMultiTask(num_classes=61, input_channels=4, pretrained=True).to(device)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n:,} params")

# ========== 5. Forward ==========
print("\n=== 5. Forward ===")
outputs = model(input_4ch)
print(f"  logits: {outputs['logits'].shape}")
print(f"  nutrition: {outputs['nutrition'].shape}")
print(f"  pred nutrition: {outputs['nutrition'].detach().cpu().numpy().tolist()}")

# ========== 6. Loss ==========
print("\n=== 6. Loss ===")
from models.multitask_losses import MultiTaskLoss
criterion = MultiTaskLoss(num_classes=61, lambda_cls=1.0, lambda_cal=1.0, lambda_weight=0.5, lambda_mape=0.1)
labels = torch.zeros(images.shape[0], dtype=torch.long).to(device)
calories = batch['calories'].to(device).float()
mass = batch['mass'].to(device).float()
nutrition_gt = torch.stack([calories, mass], dim=1)
print(f"  GT: {nutrition_gt.cpu().numpy().tolist()}")
loss, loss_dict = criterion(outputs['logits'], labels, outputs['nutrition'], nutrition_gt)
print(f"  Total loss: {loss.item():.4f}")
if loss_dict:
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")

# ========== 7. Backward + Optimizer ==========
print("\n=== 7. Backward + Optimizer ===")
loss.backward()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.step()
print("  Backward + step OK")

print("\n" + "="*50)
print("ALL CHECKS PASSED - Phase2 pipeline ready!")
print("="*50)
