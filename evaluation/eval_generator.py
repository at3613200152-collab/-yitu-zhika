"""
Phase1 生成器独立评估: PSNR / SSIM / L1 + 样本三联可视化
==========================================================
在验证集 (HSIFoodIngr-64 后 15%, 与训练完全同切分) 上:
  - PSNR: 值域 [-1,1], 10*log10(4/MSE), 与训练日志同口径 (per-image 平均)
  - SSIM : [0,1] 域, gaussian 窗口 11x11 (Wang et al. 标准, 纯 torch 实现, 零额外依赖)
  - L1   : [-1,1] 域
  - 可视化: RGB 输入 | 生成 NIR | 真实 NIR | 差异图, 保存为一张大图

用法:
    cd C:\\Users\\user\\Desktop\\yitu-zhika\\yitu-zhika-code
    C:\\Users\\user\\miniconda3\\envs\\yitu\\python.exe src\\evaluation\\eval_generator.py
    C:\\Users\\user\\miniconda3\\envs\\yitu\\python.exe src\\evaluation\\eval_generator.py --checkpoint checkpoints\\phase1\\final_model.pth --num_samples 8

输出:
    results/phase1_eval/metrics.txt        汇总指标 + 每样本明细
    results/phase1_eval/samples_grid.png   三联可视化大图
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# 路径设置 (与 train_generator.py 保持一致)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.nir_generator import UNetGenerator
from data.hsi_dataset import HSIFoodIngrDataset


# ═════════════════════ SSIM (纯 torch 实现) ═════════════════════

def gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """标准高斯窗口 [1,1,size,size], 用于 SSIM 滑动统计。"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window[None, None, :, :].contiguous()


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor,
               data_range: float = 1.0) -> float:
    """单张图 SSIM。pred/target: [1,1,H,W], 值域 [0,1]。"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1 = torch.nn.functional.conv2d(pred, window)
    mu2 = torch.nn.functional.conv2d(target, window)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ═════════════════════ 参数 ═════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase1 生成器评估: PSNR/SSIM/L1 + 可视化")
    p.add_argument("--checkpoint", type=str, default="checkpoints/phase1/final_model.pth")
    p.add_argument("--data_dir", type=str, default="data/HSIFoodIngr-64")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=8, help="可视化样本数 (取验证集前 N 个)")
    p.add_argument("--out_dir", type=str, default="results/phase1_eval")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# ═════════════════════ 评估 ═════════════════════

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA 不可用, 使用 CPU")

    # ── 路径解析 ──
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(PROJECT_ROOT, args.checkpoint)
    data_root = args.data_dir if os.path.isabs(args.data_dir) \
        else os.path.join(PROJECT_ROOT, args.data_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── 模型 ──
    print(f"\n📦 加载模型: {ckpt_path}")
    netG = UNetGenerator(in_channels=3, out_channels=1, base_channels=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt["G_state_dict"])
    netG.eval()
    print(f"  epoch: {ckpt.get('epoch', '?')}  |  G params: {sum(p.numel() for p in netG.parameters()):,}")

    # ── 验证集 (与训练完全同切分: 排序后取后 15%) ──
    val_set = HSIFoodIngrDataset(data_root, split="val", img_size=args.img_size)
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    print(f"  验证集: {len(val_set)} samples")

    # ── 逐样本评估 ──
    window = gaussian_window().to(device)
    psnrs, ssims, l1s = [], [], []
    vis = []  # (rgb, fake, real) CPU tensors [C,H,W]

    print("\n📊 逐样本评估:")
    with torch.no_grad():
        for i, (rgb, nir) in enumerate(loader):
            rgb, nir = rgb.to(device), nir.to(device)
            fake = netG(rgb)

            # PSNR: 与训练同口径 ([-1,1] 域, max=4)
            mse = torch.mean((fake - nir) ** 2).item()
            psnr = 10 * np.log10(4.0 / mse) if mse > 1e-12 else 99.0

            # SSIM: [0,1] 域
            s = ssim_torch((fake.clamp(-1, 1) + 1) / 2, (nir + 1) / 2, window)

            # L1: [-1,1] 域 (与训练日志 val/L1 同口径)
            l1 = torch.mean(torch.abs(fake - nir)).item()

            psnrs.append(psnr)
            ssims.append(s)
            l1s.append(l1)
            if len(vis) < args.num_samples:
                vis.append((rgb[0].cpu(), fake[0].clamp(-1, 1).cpu(), nir[0].cpu()))
            print(f"  [{i+1:2d}/{len(val_set)}]  PSNR={psnr:6.2f}dB  SSIM={s:.4f}  L1={l1:.4f}")

    # ── 汇总 ──
    psnrs, ssims, l1s = np.array(psnrs), np.array(ssims), np.array(l1s)
    lines = []
    lines.append("═" * 60)
    lines.append("Phase1 RGB→NIR 生成器 独立评估报告")
    lines.append("═" * 60)
    lines.append(f"Checkpoint : {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    lines.append(f"验证集     : {len(val_set)} samples  (HSIFoodIngr-64 排序后 15%, 与训练同切分)")
    lines.append(f"PSNR 口径  : 值域[-1,1], 10·log10(4/MSE), per-image 平均 (与训练日志同量纲)")
    lines.append("─" * 60)
    lines.append(f"PSNR : mean={psnrs.mean():.2f} dB  std={psnrs.std():.2f}  "
                 f"min={psnrs.min():.2f}  max={psnrs.max():.2f}")
    lines.append(f"SSIM : mean={ssims.mean():.4f}  std={ssims.std():.4f}  "
                 f"min={ssims.min():.4f}  max={ssims.max():.4f}   ([0,1]域, gaussian 11x11)")
    lines.append(f"L1   : mean={l1s.mean():.4f}  ([-1,1]域, 与训练日志 val/L1 同口径)")
    lines.append("─" * 60)
    lines.append(f"训练日志参考: Epoch 200  Val L1=0.1711  Val PSNR=18.91dB (batch 平均口径)")
    lines.append("─" * 60)
    lines.append("Per-sample 明细:")
    for i in range(len(psnrs)):
        lines.append(f"  [{i+1:2d}]  PSNR={psnrs[i]:6.2f}dB  SSIM={ssims[i]:.4f}  L1={l1s[i]:.4f}")
    report = "\n".join(lines)
    print("\n" + report)

    metrics_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n💾 指标已保存: {metrics_path}")

    # ── 三联可视化 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 未安装, 跳过可视化。安装命令:")
        print("    C:\\Users\\user\\miniconda3\\envs\\yitu\\python.exe -m pip install matplotlib")
        return

    n = len(vis)
    if n == 0:
        print("⚠️  无可视化样本")
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n), squeeze=False)

    for row, (rgb, fake, real) in enumerate(vis):
        rgb01 = ((rgb + 1) / 2).permute(1, 2, 0).numpy()          # [H,W,3]
        fake01 = ((fake + 1) / 2).squeeze(0).numpy()               # [H,W]
        real01 = ((real + 1) / 2).squeeze(0).numpy()               # [H,W]
        diff = np.abs(fake01 - real01)                             # [H,W] in [0,1]

        axes[row][0].imshow(rgb01)
        axes[row][1].imshow(fake01, cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(real01, cmap="gray", vmin=0, vmax=1)
        im = axes[row][3].imshow(diff, cmap="hot", vmin=0, vmax=0.5)

        # 行首标注该样本指标
        axes[row][0].set_ylabel(f"Sample {row+1}\nPSNR={psnrs[row]:.1f}dB\nSSIM={ssims[row]:.3f}",
                                fontsize=10)
        if row == 0:
            for c, title in enumerate(["RGB Input", "Generated NIR", "Real NIR", "|Diff| (hot)"]):
                axes[row][c].set_title(title, fontsize=12, fontweight="bold")
        for c in range(4):
            axes[row][c].set_xticks([])
            axes[row][c].set_yticks([])

    fig.colorbar(im, ax=axes[:, 3].tolist(), fraction=0.03, pad=0.02, label="|Gen - Real|")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "samples_grid.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 可视化大图已保存: {png_path}  ({n} samples × 4 视图)")
    print("\n✅ 评估完成")


if __name__ == "__main__":
    main()
