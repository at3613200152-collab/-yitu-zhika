import os
import sys
import argparse
import numpy as np
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
)
from data.dataset import get_nutrition5k_dataloaders
from models.multitask_net import MultiTaskResNet
from models.nir_generator import UNetGenerator
from evaluation.metrics import calculate_mape, calculate_rmse, calculate_accuracy


def train_one_epoch(model, train_loader, optimizer, epoch, device, writer, config):
    model.train()
    
    total_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    cal_loss_meter = AverageMeter()
    weight_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]",
                file=sys.stderr, dynamic_ncols=True)
    for i, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        predictions = model(inputs)
        losses = model.compute_loss(predictions, labels)
        
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        total_loss_meter.update(losses["total"].item(), batch_size)
        cls_loss_meter.update(losses["cls"].item(), batch_size)
        cal_loss_meter.update(losses["calories"].item(), batch_size)
        weight_loss_meter.update(losses["weight"].item(), batch_size)
        
        acc = calculate_accuracy(predictions["cls_logits"], labels["class"])
        acc_meter.update(acc, batch_size)
        
        pbar.set_postfix({
            "Loss": f"{total_loss_meter.avg:.4f}",
            "Acc": f"{acc_meter.avg:.2f}%",
            "Cal_L": f"{cal_loss_meter.avg:.2f}",
        })
        
        global_step = epoch * len(train_loader) + i
        if i % config["training"]["multitask"]["log_interval"] == 0:
            writer.add_scalar("Train/Total_loss", losses["total"].item(), global_step)
            writer.add_scalar("Train/Cls_loss", losses["cls"].item(), global_step)
            writer.add_scalar("Train/Cal_loss", losses["calories"].item(), global_step)
            writer.add_scalar("Train/Weight_loss", losses["weight"].item(), global_step)
            writer.add_scalar("Train/Accuracy", acc, global_step)
    
    return {
        "total_loss": total_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "cal_loss": cal_loss_meter.avg,
        "weight_loss": weight_loss_meter.avg,
        "accuracy": acc_meter.avg,
    }


@torch.no_grad()
def validate(model, val_loader, epoch, device, writer, log_target=True):
    model.eval()

    all_pred_cal = []
    all_true_cal = []
    all_pred_weight = []
    all_true_weight = []
    all_pred_cls = []
    all_true_cls = []

    total_loss_meter = AverageMeter()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]",
                file=sys.stderr, dynamic_ncols=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        predictions = model(inputs)
        losses = model.compute_loss(predictions, labels)

        batch_size = inputs.size(0)
        total_loss_meter.update(losses["total"].item(), batch_size)

        pred_cal = predictions["calories"].cpu().numpy()
        pred_wt = predictions["weight"].cpu().numpy()
        true_cal = labels["calories"].cpu().numpy()
        true_wt = labels["weight"].cpu().numpy()

        # 如果训练用了 log1p, 这里还原成真实 kcal / g, 这样 MAPE/RMSE 是可解释的
        if log_target:
            pred_cal = np.expm1(pred_cal)
            pred_wt = np.expm1(pred_wt)
            true_cal = np.expm1(true_cal)
            true_wt = np.expm1(true_wt)

        all_pred_cal.append(pred_cal)
        all_true_cal.append(true_cal)
        all_pred_weight.append(pred_wt)
        all_true_weight.append(true_wt)
        all_pred_cls.append(predictions["cls_logits"].cpu().numpy())
        all_true_cls.append(labels["class"].cpu().numpy())

        pbar.set_postfix({
            "Loss": f"{total_loss_meter.avg:.4f}",
        })

    all_pred_cal = np.concatenate(all_pred_cal)
    all_true_cal = np.concatenate(all_true_cal)
    all_pred_weight = np.concatenate(all_pred_weight)
    all_true_weight = np.concatenate(all_true_weight)
    all_pred_cls = np.concatenate(all_pred_cls)
    all_true_cls = np.concatenate(all_true_cls)

    cal_mape = calculate_mape(all_pred_cal, all_true_cal)
    cal_rmse = calculate_rmse(all_pred_cal, all_true_cal)
    weight_mape = calculate_mape(all_pred_weight, all_true_weight)
    weight_rmse = calculate_rmse(all_pred_weight, all_true_weight)
    accuracy = calculate_accuracy(all_pred_cls, all_true_cls)

    writer.add_scalar("Val/Total_loss", total_loss_meter.avg, epoch)
    writer.add_scalar("Val/Calories_MAPE", cal_mape, epoch)
    writer.add_scalar("Val/Calories_RMSE", cal_rmse, epoch)
    writer.add_scalar("Val/Weight_MAPE", weight_mape, epoch)
    writer.add_scalar("Val/Weight_RMSE", weight_rmse, epoch)
    writer.add_scalar("Val/Accuracy", accuracy, epoch)

    print(f"  Calories: MAPE={cal_mape:.2f}%, RMSE={cal_rmse:.2f} kcal")
    print(f"  Weight:   MAPE={weight_mape:.2f}%, RMSE={weight_rmse:.2f} g")
    print(f"  Accuracy: {accuracy:.2f}%")

    return {
        "total_loss": total_loss_meter.avg,
        "cal_mape": cal_mape,
        "cal_rmse": cal_rmse,
        "weight_mape": weight_mape,
        "weight_rmse": weight_rmse,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MultiTask Network")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--nir_ckpt", type=str, default=None, help="Path to NIR generator checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--baseline", action="store_true", help="Train baseline model (no NIR)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(42)
    
    device = torch.device(config["training"]["multitask"]["device"] 
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    nir_generator = None
    use_predicted_nir = not args.baseline
    
    if use_predicted_nir and args.nir_ckpt and os.path.exists(args.nir_ckpt):
        print(f"Loading NIR generator from {args.nir_ckpt}")
        nir_generator = UNetGenerator(
            in_channels=config["models"]["nir_generator"]["in_channels"],
            out_channels=config["models"]["nir_generator"]["out_channels"],
            base_channels=config["models"]["nir_generator"]["base_channels"],
            num_downs=config["models"]["nir_generator"]["num_downs"],
        ).to(device)
        
        checkpoint = torch.load(args.nir_ckpt, map_location=device, weights_only=False)
        if "generator_state_dict" in checkpoint:
            nir_generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            nir_generator.load_state_dict(checkpoint["model_state_dict"])
        nir_generator.eval()
        print("NIR generator loaded successfully")
    elif use_predicted_nir:
        print("Warning: No NIR generator checkpoint provided, using dummy NIR")
    
    train_loader, val_loader = get_nutrition5k_dataloaders(
        config, use_predicted_nir=use_predicted_nir, 
        nir_generator=nir_generator, device=device
    )
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    if args.baseline:
        from models.baseline import BaselineResNet
        model = BaselineResNet(config).to(device)
        model_name = "baseline"
    else:
        model = MultiTaskResNet(config).to(device)
        model_name = "multitask"
    
    model_cfg = config["models"]["multitask"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["multitask"]["epochs"]
    )
    
    start_epoch = 0
    best_cal_mape = float("inf")
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_cal_mape = checkpoint.get("best_cal_mape", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best Cal MAPE: {best_cal_mape:.2f}%")
    
    log_dir = os.path.join(config["output"]["logs_dir"], model_name)
    ckpt_dir = os.path.join(config["output"]["checkpoints_dir"], model_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    epochs = config["training"]["multitask"]["epochs"]
    save_interval = config["training"]["multitask"]["save_interval"]
    
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, epoch, device, writer, config
        )

        val_metrics = validate(
            model, val_loader, epoch, device, writer,
            log_target=config["data"]["nutrition5k"].get("log_target", True),
        )
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_metrics['total_loss']:.4f}, "
              f"Acc={train_metrics['accuracy']:.2f}% | "
              f"Val Cal MAPE={val_metrics['cal_mape']:.2f}%")
        
        if val_metrics["cal_mape"] < best_cal_mape:
            best_cal_mape = val_metrics["cal_mape"]
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_cal_mape": best_cal_mape,
                **val_metrics,
            }, best_path)
            print(f"Saved best model with Cal MAPE={best_cal_mape:.2f}%")
        
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_cal_mape": best_cal_mape,
                **val_metrics,
            }, ckpt_path)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
