"""
评估已训练模型：
1. NIR 生成器：在 HSIFoodIngr-64 验证集上计算 PSNR / SSIM
2. 多任务网络：在 Nutrition5k 真实数据验证集上计算 MAPE / RMSE / Accuracy
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from models.nir_generator import UNetGenerator
from models.multitask_net import MultiTaskResNet
from evaluation.image_metrics import ImageMetrics
from data.dataset import HSIFoodIngrDataset, Nutrition5kDataset
import yaml

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def eval_nir_generator(config):
    """评估 NIR 生成器 PSNR/SSIM"""
    print("\n" + "=" * 60)
    print("  NIR 生成器评估 (HSIFoodIngr-64 验证集)")
    print("=" * 60)

    device = torch.device("cpu")

    # 加载模型
    model = UNetGenerator(
        in_channels=3,
        out_channels=1,
        base_channels=64,
    )
    ckpt_path = "checkpoints/nir_generator/best.pth"
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.to(device)
    model.eval()

    # 加载验证集
    data_cfg = config["data"]["hsifoodingr64"]
    val_dataset = HSIFoodIngrDataset(
        rgb_dir=data_cfg["rgb_dir"],
        nir_dir=data_cfg["nir_dir"],
        image_size=data_cfg["image_size"],
        mode="val",
        train_ratio=data_cfg["train_ratio"],
    )
    print(f"  验证集: {len(val_dataset)} 样本")

    metrics = ImageMetrics(device)
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            rgb, nir_target = val_dataset[i]
            rgb_input = rgb.unsqueeze(0).to(device)
            # Normalize to [-1, 1] for generator
            rgb_norm = (rgb_input - 0.5) / 0.5
            pred_nir = model(rgb_norm)
            # Normalize target
            nir_target = nir_target.unsqueeze(0).to(device)
            nir_norm = (nir_target - 0.5) / 0.5

            result = metrics.compute(pred_nir, nir_norm, max_val=1.0)
            psnr_list.append(result["psnr"])
            ssim_list.append(result["ssim"])

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(val_dataset)}  PSNR={np.mean(psnr_list):.2f}  SSIM={np.mean(ssim_list):.4f}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)

    print(f"\n  --- NIR 生成器结果 ---")
    print(f"  PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    return {"psnr": avg_psnr, "ssim": avg_ssim}

def eval_multitask(config):
    """评估多任务网络"""
    print("\n" + "=" * 60)
    print("  多任务网络评估 (Nutrition5k 真实数据验证集)")
    print("=" * 60)

    device = torch.device("cpu")

    # 加载 NIR 生成器（用于预测 NIR 通道）
    nir_gen = UNetGenerator(in_channels=3, out_channels=1, base_channels=64)
    nir_ckpt = "checkpoints/nir_generator/best.pth"
    if os.path.exists(nir_ckpt):
        ckpt = torch.load(nir_ckpt, weights_only=False, map_location=device)
        nir_gen.load_state_dict(ckpt["generator_state_dict"])
        nir_gen.to(device)
        nir_gen.eval()
        print("  NIR 生成器: 已加载")
    else:
        nir_gen = None
        print("  NIR 生成器: 未找到，使用零填充")

    # 加载多任务模型
    mt_cfg = config["models"]["multitask"]
    model = MultiTaskResNet(config)
    mt_ckpt = "checkpoints/multitask/best.pth"
    if not os.path.exists(mt_ckpt):
        print(f"  [SKIP] checkpoint not found: {mt_ckpt}")
        return None

    ckpt = torch.load(mt_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    mape = ckpt.get("best_cal_mape", "?")
    print(f"  多任务网络: 已加载 (Epoch {epoch}, Best MAPE={mape})")

    # 加载验证集
    data_cfg = config["data"]["nutrition5k"]
    val_dataset = Nutrition5kDataset(
        rgb_dir=data_cfg["rgb_dir"],
        labels_file=data_cfg["labels_file"],
        image_size=data_cfg["image_size"],
        mode="val",
        train_ratio=data_cfg["train_ratio"],
        use_predicted_nir=True,
        nir_generator=nir_gen,
        device=device,
    )
    print(f"  验证集: {len(val_dataset)} 样本")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    all_cal_pred, all_cal_true = [], []
    all_wt_pred, all_wt_true = [], []
    all_cls_pred, all_cls_true = [], []
    total_loss = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="  Evaluating"):
            inputs = inputs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            preds = model(inputs)
            losses = model.compute_loss(preds, labels)
            total_loss += losses["total"].item()
            count += 1

            all_cal_pred.extend(preds["calories"].cpu().numpy())
            all_cal_true.extend(labels["calories"].cpu().numpy())
            all_wt_pred.extend(preds["weight"].cpu().numpy())
            all_wt_true.extend(labels["weight"].cpu().numpy())
            all_cls_pred.extend(preds["cls_logits"].argmax(dim=1).cpu().numpy())
            all_cls_true.extend(labels["class"].cpu().numpy())

    all_cal_pred = np.array(all_cal_pred)
    all_cal_true = np.array(all_cal_true)
    all_wt_pred = np.array(all_wt_pred)
    all_wt_true = np.array(all_wt_true)
    all_cls_pred = np.array(all_cls_pred)
    all_cls_true = np.array(all_cls_true)

    # 指标
    cal_mape = np.mean(np.abs(all_cal_pred - all_cal_true) / np.clip(np.abs(all_cal_true), 1, None)) * 100
    cal_rmse = np.sqrt(np.mean((all_cal_pred - all_cal_true) ** 2))
    wt_mape = np.mean(np.abs(all_wt_pred - all_wt_true) / np.clip(np.abs(all_wt_true), 1, None)) * 100
    wt_rmse = np.sqrt(np.mean((all_wt_pred - all_wt_true) ** 2))
    acc = (all_cls_pred == all_cls_true).mean() * 100

    print(f"\n  --- 多任务网络结果 ---")
    print(f"  Calories:  MAPE={cal_mape:.2f}%  RMSE={cal_rmse:.2f}")
    print(f"  Weight:    MAPE={wt_mape:.2f}%  RMSE={wt_rmse:.2f}")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Avg Loss:  {total_loss/count:.4f}")

    return {"cal_mape": cal_mape, "cal_rmse": cal_rmse, "wt_mape": wt_mape, "wt_rmse": wt_rmse, "acc": acc}

if __name__ == "__main__":
    config = load_config()
    nir_result = eval_nir_generator(config)
    mt_result = eval_multitask(config)

    print("\n" + "=" * 60)
    print("  总结")
    print("=" * 60)
    if nir_result:
        print(f"  NIR 生成器:  PSNR={nir_result['psnr']:.2f} dB  SSIM={nir_result['ssim']:.4f}")
    if mt_result:
        print(f"  多任务网络:  Cal MAPE={mt_result['cal_mape']:.2f}%  RMSE={mt_result['cal_rmse']:.2f}")
        print(f"              Acc={mt_result['acc']:.2f}%")
