import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    load_config,
    set_seed,
    save_checkpoint,
    AverageMeter,
    save_image,
    plot_comparison,
)
from data.dataset import get_hsi_dataloaders
from models.nir_generator import Pix2PixModel, UNetGenerator
from evaluation.metrics import calculate_psnr, calculate_ssim


def train_one_epoch(model, train_loader, optimizer_g, optimizer_d, epoch, device, writer, config):
    model.train()
    
    loss_g_meter = AverageMeter()
    loss_d_meter = AverageMeter()
    loss_gan_meter = AverageMeter()
    loss_l1_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for i, (rgb, nir) in enumerate(pbar):
        rgb = rgb.to(device)
        nir = nir.to(device)
        
        fake_nir = model(rgb)
        
        optimizer_d.zero_grad()
        loss_d = model.compute_discriminator_loss(rgb, nir, fake_nir)
        loss_d.backward()
        optimizer_d.step()
        
        fake_nir = model(rgb)
        optimizer_g.zero_grad()
        loss_g, loss_gan, loss_l1 = model.compute_generator_loss(rgb, nir, fake_nir)
        loss_g.backward()
        optimizer_g.step()
        
        loss_g_meter.update(loss_g.item(), rgb.size(0))
        loss_d_meter.update(loss_d.item(), rgb.size(0))
        loss_gan_meter.update(loss_gan.item(), rgb.size(0))
        loss_l1_meter.update(loss_l1.item(), rgb.size(0))
        
        pbar.set_postfix({
            "G_loss": f"{loss_g_meter.avg:.4f}",
            "D_loss": f"{loss_d_meter.avg:.4f}",
            "L1": f"{loss_l1_meter.avg:.4f}",
        })
        
        global_step = epoch * len(train_loader) + i
        if i % config["training"]["nir_generator"]["log_interval"] == 0:
            writer.add_scalar("Train/G_loss", loss_g.item(), global_step)
            writer.add_scalar("Train/D_loss", loss_d.item(), global_step)
            writer.add_scalar("Train/GAN_loss", loss_gan.item(), global_step)
            writer.add_scalar("Train/L1_loss", loss_l1.item(), global_step)
    
    return {
        "loss_g": loss_g_meter.avg,
        "loss_d": loss_d_meter.avg,
        "loss_gan": loss_gan_meter.avg,
        "loss_l1": loss_l1_meter.avg,
    }


@torch.no_grad()
def validate(model, val_loader, epoch, device, writer, config, save_dir):
    model.eval()
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    loss_l1_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    for i, (rgb, nir) in enumerate(pbar):
        rgb = rgb.to(device)
        nir = nir.to(device)
        
        fake_nir = model(rgb)
        
        l1_loss = nn.L1Loss()(fake_nir, nir)
        loss_l1_meter.update(l1_loss.item(), rgb.size(0))
        
        for b in range(rgb.size(0)):
            psnr = calculate_psnr(fake_nir[b:b+1], nir[b:b+1], data_range=2.0)
            ssim = calculate_ssim(fake_nir[b:b+1], nir[b:b+1], data_range=2.0)
            psnr_meter.update(psnr, 1)
            ssim_meter.update(ssim, 1)
        
        pbar.set_postfix({
            "PSNR": f"{psnr_meter.avg:.2f}",
            "SSIM": f"{ssim_meter.avg:.4f}",
            "L1": f"{loss_l1_meter.avg:.4f}",
        })
        
        if i == 0:
            for b in range(min(4, rgb.size(0))):
                save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{b}.png")
                plot_comparison(rgb[b], nir[b], fake_nir[b], save_path)
    
    writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
    writer.add_scalar("Val/SSIM", ssim_meter.avg, epoch)
    writer.add_scalar("Val/L1_loss", loss_l1_meter.avg, epoch)
    
    return {
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
        "l1_loss": loss_l1_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Train NIR Generator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(42)
    
    device = torch.device(config["training"]["nir_generator"]["device"] 
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_hsi_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    model = Pix2PixModel(config).to(device)
    
    model_cfg = config["models"]["nir_generator"]
    optimizer_g = torch.optim.Adam(
        model.generator.parameters(),
        lr=model_cfg["lr"],
        betas=(model_cfg["beta1"], model_cfg["beta2"]),
    )
    optimizer_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=model_cfg["lr"],
        betas=(model_cfg["beta1"], model_cfg["beta2"]),
    )
    
    start_epoch = 0
    best_psnr = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.generator.load_state_dict(checkpoint["generator_state_dict"])
        model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", 0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    log_dir = os.path.join(config["output"]["logs_dir"], "nir_generator")
    ckpt_dir = os.path.join(config["output"]["checkpoints_dir"], "nir_generator")
    vis_dir = os.path.join(config["output"]["results_dir"], "nir_generator_vis")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    epochs = config["training"]["nir_generator"]["epochs"]
    save_interval = config["training"]["nir_generator"]["save_interval"]
    
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer_g, optimizer_d, epoch, device, writer, config
        )
        
        val_metrics = validate(
            model, val_loader, epoch, device, writer, config, vis_dir
        )
        
        print(f"Epoch {epoch}: Train G_loss={train_metrics['loss_g']:.4f}, "
              f"D_loss={train_metrics['loss_d']:.4f} | "
              f"Val PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}")
        
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "generator_state_dict": model.generator.state_dict(),
                "discriminator_state_dict": model.discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "best_psnr": best_psnr,
                "psnr": val_metrics["psnr"],
                "ssim": val_metrics["ssim"],
            }, best_path)
            print(f"Saved best model with PSNR={best_psnr:.2f}")
        
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "generator_state_dict": model.generator.state_dict(),
                "discriminator_state_dict": model.discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "best_psnr": best_psnr,
                "psnr": val_metrics["psnr"],
                "ssim": val_metrics["ssim"],
            }, ckpt_path)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
