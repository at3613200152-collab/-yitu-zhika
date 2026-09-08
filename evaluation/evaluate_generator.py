"""
阶段一评估脚本: NIR生成器定量评估 + 三联图可视化
====================================================
加载 checkpoints/generator/best_model.pt，在验证集(或测试集)上计算
逐图 PSNR / SSIM，并保存 RGB | 真实NIR | 预测NIR 三联图。

用法 (在 D:\\-yitu-zhika 下运行):
    & .venv\\Scripts\\python.exe evaluation\\evaluate_generator.py
    & .venv\\Scripts\\python.exe evaluation\\evaluate_generator.py --checkpoint checkpoints\\generator\\checkpoint_epoch_0199.pt
    & .venv\\Scripts\\python.exe evaluation\\evaluate_generator.py --split test --num-images 12

输出 (output/generator_eval/):
    eval_summary.txt        汇总指标(总体 + 分类别 PSNR)
    triplet_0000.png ...    前 N 张三联图
    triplet_best.png        PSNR 最高样本
    triplet_worst.png       PSNR 最低样本

注意: 本脚本才是真实评估。evaluation/image_metrics.py 只是指标函数库，
其 __main__ 里的"高质量/低质量预测"是对随机噪声的自检，与模型成绩无关。
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image as PILImage, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.generator.pix2pix import Pix2PixModel
from data.hsifoodingr_loader import create_hsifoodingr_dataloader
from evaluation.image_metrics import psnr, ssim


def parse_args():
    parser = argparse.ArgumentParser(description="阶段一: 评估NIR生成器 (PSNR/SSIM + 三联图)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/generator/best_model.pt")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--num-images", type=int, default=8,
                        help="保存前N张三联图")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_model(ckpt_path: str, gen_cfg: dict, device: torch.device) -> Pix2PixModel:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到checkpoint: {ckpt_path}")
    model = Pix2PixModel(
        input_channels=gen_cfg.get("input_channels", 3),
        output_channels=gen_cfg.get("output_channels", 1),
        base_features_g=gen_cfg.get("base_features", 64),
        lambda_l1=gen_cfg.get("lambda_l1", 100.0),
        lambda_gan=gen_cfg.get("lambda_gan", 1.0),
    )
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    print(f"已加载: {ckpt_path} (epoch={epoch})")
    return model


def save_triplet(path: str, rgb01: torch.Tensor, nir01: torch.Tensor,
                 pred01: torch.Tensor) -> None:
    """保存 RGB | 真实NIR | 预测NIR 三联图。输入均为 [0,1] 域张量。"""
    H, W = rgb01.shape[-2], rgb01.shape[-1]
    bar = 26
    canvas = PILImage.new("RGB", (W * 3, H + bar), (24, 24, 24))

    rgb_u8 = (rgb01.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    nir_u8 = np.repeat((nir01[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., None], 3, axis=-1)
    pred_u8 = np.repeat((pred01[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., None], 3, axis=-1)

    canvas.paste(PILImage.fromarray(rgb_u8), (0, bar))
    canvas.paste(PILImage.fromarray(nir_u8), (W, bar))
    canvas.paste(PILImage.fromarray(pred_u8), (W * 2, bar))

    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(["RGB", "Real NIR", "Pred NIR"]):
        draw.text((i * W + 8, 7), text, fill=(255, 255, 255))

    canvas.save(path)


def forward_single(model, sample, device):
    rgb = sample["rgb"].unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        pred = model.generator(rgb)
    return sample["rgb"], sample["nir"], ((pred + 1.0) / 2.0)[0].cpu()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    gen_cfg = config.get("generator", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据
    data_path = data_cfg.get("hsifoodingr_path", "./data/HSIFoodIngr-64")
    img_size = data_cfg.get("img_size", 256)
    val_loader = create_hsifoodingr_dataloader(
        root_dir=data_path, split=args.split, img_size=img_size,
        batch_size=args.batch_size, num_workers=data_cfg.get("num_workers", 4),
        augmentation=False,
    )
    dataset = val_loader.dataset
    class_names = dataset.class_names
    print(f"{args.split}集: {len(dataset)} 样本, {dataset.num_classes} 类")

    # 模型
    model = load_model(args.checkpoint, gen_cfg, device)

    # 评估
    per_psnr, per_ssim = [], []
    batch_psnr_list = []          # 训练日志口径: 整batch MSE -> PSNR, 再对batch取平均
    class_psnr = {}               # 类别 -> [psnr列表]
    saved = 0
    global_idx = 0
    best = (float("-inf"), None)  # (psnr, global_idx)
    worst = (float("inf"), None)

    out_dir = os.path.join(config.get("output_dir", "./output"), "generator_eval")
    os.makedirs(out_dir, exist_ok=True)

    print("评估中...")
    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device) * 2.0 - 1.0
            nir_real = batch["nir"].to(device)
            nir_pred = model.generator(rgb)
            nir_pred01 = (nir_pred + 1.0) / 2.0

            # 训练日志口径(整batch)
            mse = torch.mean((nir_pred01 - nir_real) ** 2)
            if mse > 0:
                batch_psnr_list.append(10 * torch.log10(1.0 / mse).item())

            # 逐图口径
            B = nir_real.shape[0]
            for i in range(B):
                p = psnr(nir_pred01[i:i + 1], nir_real[i:i + 1]).item()
                s = ssim(nir_pred01[i:i + 1], nir_real[i:i + 1]).item()
                per_psnr.append(p)
                per_ssim.append(s)
                label = int(batch["label"][i])
                class_psnr.setdefault(label, []).append(p)

                if p > best[0]:
                    best = (p, global_idx)
                if p < worst[0]:
                    worst = (p, global_idx)

                # 保存前N张三联图
                if saved < args.num_images:
                    save_triplet(
                        os.path.join(out_dir, f"triplet_{global_idx:04d}.png"),
                        batch["rgb"][i], nir_real[i].cpu(), nir_pred01[i].cpu(),
                    )
                    saved += 1
                global_idx += 1

    # 汇总
    per_psnr_np = np.array(per_psnr)
    per_ssim_np = np.array(per_ssim)
    train_style = float(np.mean(batch_psnr_list)) if batch_psnr_list else float("nan")

    lines = []
    lines.append("=" * 60)
    lines.append(f"NIR生成器评估  split={args.split}  checkpoint={args.checkpoint}")
    lines.append("=" * 60)
    lines.append(f"样本数: {len(per_psnr)}")
    lines.append("")
    lines.append("[逐图口径] 与论文可比的报法")
    lines.append(f"  PSNR: mean={per_psnr_np.mean():.2f} dB, std={per_psnr_np.std():.2f}, "
                 f"median={np.median(per_psnr_np):.2f}, min={per_psnr_np.min():.2f}, max={per_psnr_np.max():.2f}")
    lines.append(f"  SSIM: mean={per_ssim_np.mean():.4f}, std={per_ssim_np.std():.4f}")
    lines.append("")
    lines.append(f"[训练日志口径] 整batch计算后取平均: PSNR={train_style:.2f} dB")
    lines.append("")
    lines.append("[分类别 PSNR] (样本数, 平均PSNR)")
    for label in sorted(class_psnr.keys()):
        arr = np.array(class_psnr[label])
        name = class_names[label] if 0 <= label < len(class_names) else "?"
        lines.append(f"  {name:<20s} n={len(arr):3d}  PSNR={arr.mean():.2f} dB")
    lines.append("")
    lines.append(f"最佳样本: idx={best[1]}, PSNR={best[0]:.2f} dB -> triplet_best.png")
    lines.append(f"最差样本: idx={worst[1]}, PSNR={worst[0]:.2f} dB -> triplet_worst.png")

    # best / worst 三联图
    if best[1] is not None:
        sample = dataset[best[1]]
        r, n, pr = forward_single(model, sample, device)
        save_triplet(os.path.join(out_dir, "triplet_best.png"), r, n, pr)
    if worst[1] is not None:
        sample = dataset[worst[1]]
        r, n, pr = forward_single(model, sample, device)
        save_triplet(os.path.join(out_dir, "triplet_worst.png"), r, n, pr)

    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print()
    print(summary)
    print()
    print(f"汇总已保存: {os.path.join(out_dir, 'eval_summary.txt')}")
    print(f"三联图目录: {out_dir}")


if __name__ == "__main__":
    main()
