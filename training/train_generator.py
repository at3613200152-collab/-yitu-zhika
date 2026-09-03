"""
Phase1 训练脚本: RGB→NIR 生成器 (Pix2Pix GAN)
================================================
训练 U-Net 生成器从 RGB 图像生成 NIR 图像，使用 PatchGAN 判别器。

使用方法:
    python src/training/train_generator.py
    python src/training/train_generator.py --epochs 200 --batch_size 4
    python src/training/train_generator.py --resume
"""

import os
import sys
import time
import argparse
import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 路径设置
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.nir_generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from data.pix2pix_dataset import build_dataloaders


# ═════════════════════ LR 调度 ═════════════════════
def get_lr_lambda(total_epochs, decay_start):
    def lr_lambda(epoch):
        if epoch + 1 <= decay_start:
            return 1.0
        decay_len = max(1, total_epochs - decay_start)
        return max(0.0, 1.0 - (epoch + 1 - decay_start) / decay_len)
    return lr_lambda


# ═════════════════════ Checkpoint ═════════════════════
def save_checkpoint(path, epoch, netG, netD, optG, optD, schedG, schedD, loss_G, loss_D):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch, "G_state_dict": netG.state_dict(), "D_state_dict": netD.state_dict(),
        "optG_state_dict": optG.state_dict(), "optD_state_dict": optD.state_dict(),
        "schedG_state_dict": schedG.state_dict(), "schedD_state_dict": schedD.state_dict(),
        "loss_G": loss_G, "loss_D": loss_D,
    }, path)
    print(f"  💾 Saved: {path}")


def load_checkpoint(path, netG, netD, optG, optD, schedG, schedD, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt["G_state_dict"])
    netD.load_state_dict(ckpt["D_state_dict"])
    optG.load_state_dict(ckpt["optG_state_dict"])
    optD.load_state_dict(ckpt["optD_state_dict"])
    if "schedG_state_dict" in ckpt:
        schedG.load_state_dict(ckpt["schedG_state_dict"])
    if "schedD_state_dict" in ckpt:
        schedD.load_state_dict(ckpt["schedD_state_dict"])
    print(f"  ✅ Resumed epoch {ckpt['epoch']}")
    return ckpt["epoch"]


def find_latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    return max(files, key=os.path.getmtime) if files else None


# ═════════════════════ 参数 ═════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Phase1: RGB→NIR Generator Training")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--decay_start", type=int, default=100)
    p.add_argument("--data_dirs", type=str, nargs="+", 
                   default=["data/nirscene1_x10/nirscene_img_aug_10_oversample",
                            "data/capsicum/capsicums_pix2pixHD"])
    p.add_argument("--hsi_root", type=str, default="data/HSIFoodIngr-64")
    p.add_argument("--hsi_weight", type=float, default=5.0)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/phase1")
    p.add_argument("--log_dir", type=str, default="logs/phase1")
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--pretrained", type=str, default=None,
                   help="加载预训练generator权重 (队友PSNR24模型)")
    p.add_argument("--resume", type=str, default=None, nargs="?", const="latest")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# ═════════════════════ 训练 ═════════════════════
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  CUDA 不可用，使用 CPU")

    # 路径
    data_dirs = [d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d) for d in args.data_dirs]
    hsi_root = args.hsi_root if os.path.isabs(args.hsi_root) else os.path.join(PROJECT_ROOT, args.hsi_root)
    ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(PROJECT_ROOT, args.ckpt_dir)
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(PROJECT_ROOT, args.log_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 模型
    print("\n📦 初始化模型...")
    netG = UNetGenerator(in_channels=3, out_channels=1, base_channels=64).to(device)
    netD = PatchDiscriminator(in_channels=4, ndf=64).to(device)
    print(f"  G params: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"  D params: {sum(p.numel() for p in netD.parameters()):,}")

    # 加载预训练权重
    if args.pretrained:
        pre_path = args.pretrained if os.path.isabs(args.pretrained) else os.path.join(PROJECT_ROOT, args.pretrained)
        if os.path.exists(pre_path):
            print(f"\n📥 加载预训练权重: {pre_path}")
            pre_ckpt = torch.load(pre_path, map_location=device, weights_only=False)
            if "generator_state_dict" in pre_ckpt:
                netG.load_state_dict(pre_ckpt["generator_state_dict"])
                print(f"  ✅ Generator weights loaded (epoch {pre_ckpt.get('epoch', '?')}, PSNR {pre_ckpt.get('best_psnr', '?')})")
            elif "G_state_dict" in pre_ckpt:
                netG.load_state_dict(pre_ckpt["G_state_dict"])
                print(f"  ✅ Generator weights loaded (our format, epoch {pre_ckpt.get('epoch', '?')})")
            else:
                # 直接是 state_dict
                netG.load_state_dict(pre_ckpt)
                print(f"  ✅ Generator weights loaded (raw state_dict)")
            # 冻结encoder前几层可选，默认全部fine-tune
        else:
            print(f"  ⚠️  Pretrained file not found: {pre_path}, training from scratch")

    # 损失 & 优化器
    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    schedG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda=get_lr_lambda(args.epochs, args.decay_start))
    schedD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda=get_lr_lambda(args.epochs, args.decay_start))

    # 数据 — build_dataloaders 返回 (train_loader, val_loader)
    print(f"\n📊 数据集: {data_dirs}")
    if os.path.isdir(hsi_root):
        print(f"  HSI data: {hsi_root} (weight={args.hsi_weight})")
    train_loader, val_loader = build_dataloaders(
        data_roots=data_dirs, batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=args.img_size,
        hsi_root=hsi_root, hsi_weight=args.hsi_weight)
    print(f"  train: {len(train_loader.dataset)} samples, val: {len(val_loader.dataset)} samples")

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # 续训
    start_epoch = 0
    if args.resume:
        if args.resume == "latest":
            resume_path = find_latest_checkpoint(ckpt_dir)
        else:
            resume_path = args.resume
            if not os.path.isabs(resume_path):
                resume_path = os.path.join(PROJECT_ROOT, resume_path)
        if resume_path and os.path.exists(resume_path):
            start_epoch = load_checkpoint(resume_path, netG, netD, optG, optD, schedG, schedD, device) + 1
        else:
            print("⚠️  No checkpoint found, starting fresh")

    # 摘要
    print(f"\n{'='*60}")
    print(f"  Phase1: RGB→NIR  |  {start_epoch}→{args.epochs} epochs  |  bs={args.batch_size}  |  lr={args.lr}")
    print(f"{'='*60}\n")

    torch.backends.cudnn.benchmark = True
    avg_g = 0.0
    avg_d = 0.0

    for epoch in range(start_epoch, args.epochs):
        netG.train(); netD.train()
        t0 = time.time()
        d_losses, g_losses = [], []

        for i, (real_rgb, real_nir) in enumerate(train_loader):
            # 数据: (rgb, nir) tuple, 值域 [-1,1]
            real_rgb = real_rgb.to(device, non_blocking=True)
            real_nir = real_nir.to(device, non_blocking=True)

            # ── D step ──
            optD.zero_grad()
            with torch.no_grad():
                fake_nir = netG(real_rgb)
            pred_real = netD(real_rgb, real_nir)
            pred_fake = netD(real_rgb, fake_nir)
            loss_D = 0.5 * (criterionGAN(pred_real, torch.ones_like(pred_real)) +
                            criterionGAN(pred_fake, torch.zeros_like(pred_fake)))
            loss_D.backward(); optD.step()

            # ── G step ──
            optG.zero_grad()
            fake_nir = netG(real_rgb)
            pred_fake = netD(real_rgb, fake_nir)
            loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterionL1(fake_nir, real_nir)
            loss_G = loss_G_GAN + args.lambda_l1 * loss_G_L1
            loss_G.backward(); optG.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

            if i % 50 == 0:
                cur_lr = optG.param_groups[0]["lr"]
                print(f"  [E{epoch+1} B{i+1}] D:{loss_D.item():.4f} G:{loss_G.item():.4f} (GAN:{loss_G_GAN.item():.4f} L1:{loss_G_L1.item():.4f}) lr:{cur_lr:.6f}")
                step = epoch * len(train_loader) + i
                writer.add_scalar("train/D_loss", loss_D.item(), step)
                writer.add_scalar("train/G_loss", loss_G.item(), step)
                writer.add_scalar("train/G_GAN", loss_G_GAN.item(), step)
                writer.add_scalar("train/G_L1", loss_G_L1.item(), step)

        # Epoch 统计
        if not d_losses:
            print(f"\n  Epoch {epoch+1}/{args.epochs} — no training data, skipping")
            schedG.step(); schedD.step()
            continue
        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) D:{avg_d:.4f} G:{avg_g:.4f}")

        writer.add_scalar("epoch/D_loss", avg_d, epoch+1)
        writer.add_scalar("epoch/G_loss", avg_g, epoch+1)

        # 验证
        if val_loader is not None:
            netG.eval()
            val_l1, val_psnr, n_val = 0, 0, 0
            with torch.no_grad():
                for rgb_v, nir_v in val_loader:
                    rgb_v, nir_v = rgb_v.to(device), nir_v.to(device)
                    pred = netG(rgb_v)
                    val_l1 += criterionL1(pred, nir_v).item()
                    mse = torch.mean((pred - nir_v) ** 2)
                    if mse > 0:
                        val_psnr += (10 * torch.log10(4.0 / mse)).item()
                    n_val += 1
            avg_val_l1 = val_l1 / max(n_val, 1)
            avg_val_psnr = val_psnr / max(n_val, 1)
            print(f"  Val: L1={avg_val_l1:.4f} PSNR={avg_val_psnr:.2f}dB")
            writer.add_scalar("val/L1", avg_val_l1, epoch+1)
            writer.add_scalar("val/PSNR", avg_val_psnr, epoch+1)

        # Checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth")
            save_checkpoint(ckpt_path, epoch+1, netG, netD, optG, optD, schedG, schedD, avg_g, avg_d)
            # TensorBoard 图像
            netG.eval()
            with torch.no_grad():
                writer.add_image("RGB", (real_rgb[0].cpu() + 1) / 2, epoch+1)
                writer.add_image("NIR_real", (real_nir[0].cpu() + 1) / 2, epoch+1)
                writer.add_image("NIR_fake", (netG(real_rgb[:1])[0].cpu() + 1) / 2, epoch+1)

        schedG.step(); schedD.step()
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    writer.close()
    final_path = os.path.join(ckpt_dir, "final_model.pth")
    save_checkpoint(final_path, args.epochs, netG, netD, optG, optD, schedG, schedD, avg_g, avg_d)
    print(f"\n✅ 训练完成! Final: {final_path}")


if __name__ == "__main__":
    train(parse_args())
