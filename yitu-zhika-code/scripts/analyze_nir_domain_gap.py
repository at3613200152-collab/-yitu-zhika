#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：预测 NIR 为什么带不来增益 —— 跨库域差异（尤其是对比度）到底有多大？

回答的问题：
  生成器在 HSIFoodIngr-64（实验室高光谱相机）上训练，却作用在 Nutrition5k（手机照片）上。
  两个库的 RGB「对比度/色调」不同，这是否就是预测 NIR 无效的原因？

做法：
  1) 域 A（训练域，同分布）：data/hsi_full_v3 的 test 划分（327 张），有**实测 NIR** 可作参照；
  2) 域 B（部署域，跨库）：Nutrition5k 冻结 test 划分（507 张），走与训练完全一致的预处理；
  3) 两域都跑同一个冻结生成器（ms3 用的 full_seed42），比较：
     - RGB 输入的亮度/对比度/饱和像素比例
     - 预测 NIR 的全局均值/方差、**逐图对比度**、饱和比例、跨图方差
     - 以「域 A 的实测 NIR」作为目标分布基准，看域 B 的预测 NIR 偏了多少
  4) 对照实验：把域 B 的输入按域 A 的逐图亮度/对比度统计做**对比度对齐**后重新生成，
     看预测 NIR 的分布是否回到同分布区间（若是，说明差异主要是一阶对比度；若否，说明是结构性域差异）。

自校验：脚本会先在域 A 上复算 PSNR，应约等于 26.65 dB；不等则说明复现的预处理口径不对。

用法：
  python scripts/analyze_nir_domain_gap.py [--limit-b 507] [--out artifacts/nir-domain-gap.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.generator import UNetGenerator            # noqa: E402
from src.training.train_meal_official import MANIFEST, MEAN, STD  # noqa: E402

GEN_DEFAULT = ROOT / "checkpoints/hsi_full_v3/full_seed42/best.pt"
HSI_DIR = ROOT / "data/hsi_full_v3"


def load_generator(path: Path, device: str):
    gen = UNetGenerator(in_channels=3, out_channels=1, base_filters=64).to(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("G_state_dict") or ckpt.get("model") or ckpt.get("model_state_dict")
    if state is None:
        raise SystemExit("无法从 %s 解析生成器权重" % path)
    gen.load_state_dict(state, strict=True)
    gen.eval()
    return gen, state


@torch.no_grad()
def predict(gen, rgb01: torch.Tensor, device: str, batch: int = 32) -> np.ndarray:
    """rgb01: (N,3,H,W) in [0,1] -> (N,H,W) 预测 NIR，映射回 [0,1]。

    与 train_hsi_full_v3.py 的推理口径完全一致：bf16 autocast 前向、fp32 后处理。
    """
    outs = []
    for i in range(0, len(rgb01), batch):
        chunk = rgb01[i:i + batch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            nir = gen(chunk * 2.0 - 1.0)
        nir = nir.float()
        outs.append(((nir + 1.0) / 2.0).clamp(0, 1).squeeze(1).cpu())
    return torch.cat(outs).numpy()


def image_stats(x: np.ndarray, tag: str) -> dict:
    """x: (N,H,W) in [0,1]"""
    flat = x.reshape(len(x), -1)
    per_mean = flat.mean(1)
    per_std = flat.std(1)
    lo = np.percentile(flat, 2, axis=1)
    hi = np.percentile(flat, 98, axis=1)
    sat = ((flat <= 0.001) | (flat >= 0.999)).mean(1)
    return {
        "tag": tag,
        "n": int(len(x)),
        "global_mean": float(flat.mean()),
        "global_std": float(flat.std()),
        "per_image_mean_avg": float(per_mean.mean()),
        "per_image_contrast_avg": float(per_std.mean()),
        "per_image_contrast_median": float(np.median(per_std)),
        "per_image_span_p2_p98_avg": float((hi - lo).mean()),
        "saturated_fraction": float(sat.mean()),
        "between_image_std_of_means": float(per_mean.std()),
    }


def rgb_stats(rgb01: np.ndarray, tag: str) -> dict:
    """rgb01: (N,3,H,W) in [0,1] -> 亮度/对比度/饱和统计"""
    lum = (0.299 * rgb01[:, 0] + 0.587 * rgb01[:, 1] + 0.114 * rgb01[:, 2])
    flat = lum.reshape(len(lum), -1)
    lum_std = flat.std(1)
    per_ch_std = rgb01.reshape(len(rgb01), 3, -1).std(2)
    return {
        "tag": tag,
        "luma_mean": float(flat.mean()),
        "luma_contrast_avg": float(lum_std.mean()),
        "luma_contrast_median": float(np.median(lum_std)),
        "channel_std_avg": [float(v) for v in per_ch_std.mean(0)],
        "saturated_fraction": float(((rgb01 <= 0.002) | (rgb01 >= 0.998)).mean()),
        "dark_fraction": float((lum < 0.05).mean()),
        "bright_fraction": float((lum > 0.95).mean()),
    }


def psnr_pair(pred: np.ndarray, real: np.ndarray) -> dict:
    """与 train_hsi_full_v3.py 的 metrics() 同口径：10·log10(1/MSE)，[0,1] 域。

    返回逐图 PSNR 均值（报告口径）与批 MSE 口径 FPSNR（对照口径）。
    """
    mse_per_image = ((pred.astype(np.float64) - real.astype(np.float64)) ** 2).reshape(len(pred), -1).mean(1)
    per_image = 10.0 * np.log10(1.0 / np.maximum(mse_per_image, 1e-12))
    batch = 10.0 * np.log10(1.0 / max(float(mse_per_image.mean()), 1e-12))
    return {"psnr_per_image_mean": float(per_image.mean()),
            "psnr_batch_mse": float(batch),
            "n": int(len(pred))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default=str(GEN_DEFAULT))
    ap.add_argument("--limit-b", type=int, default=0, help="只用前 N 张 Nutrition5k test（0=全部）")
    ap.add_argument("--limit-a", type=int, default=0, help="只用前 N 张 HSI test（0=全部）")
    ap.add_argument("--out", default="artifacts/nir-domain-gap.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen, gen_state = load_generator(Path(args.generator), device)
    print("generator=%s device=%s" % (args.generator, device))

    def predict_mode(model, rgb, mode: str, batch: int) -> np.ndarray:
        """按给定 BN 模式复现推理。train 模式会改写 running stats，故每次先重载权重。"""
        model.load_state_dict(gen_state, strict=True)
        model.train(mode == "train")
        try:
            return predict(model, rgb, device, batch=batch)
        finally:
            model.eval()

    # ---------- 域 A：HSIFoodIngr-64（训练域，有实测 NIR） ----------
    rgb_a = np.load(HSI_DIR / "rgb_test.npy", mmap_mode="r")
    nir_a = np.load(HSI_DIR / "nir_test.npy", mmap_mode="r")
    n_a = len(rgb_a) if not args.limit_a else min(args.limit_a, len(rgb_a))
    rgb_a = np.asarray(rgb_a[:n_a], dtype=np.float32) / 255.0
    nir_a = np.asarray(nir_a[:n_a], dtype=np.float32)
    t_a = torch.from_numpy(rgb_a).permute(0, 3, 1, 2).contiguous()
    pred_a = predict(gen, t_a, device)                      # eval 模式：与阶段二管线一致
    ps_a = psnr_pair(pred_a, nir_a)
    pred_a_train8 = predict_mode(gen, t_a, "train", 8)      # 复现 train_hsi_full_v3 的测试阶段
    ps_a_train8 = psnr_pair(pred_a_train8, nir_a)
    print("domain A (HSI test) n=%d" % n_a)
    print("  eval  模式 batch=32: 逐图 %.3f dB / 批MSE %.3f dB   <- 阶段二实际喂给 ResNet 的口径"
          % (ps_a["psnr_per_image_mean"], ps_a["psnr_batch_mse"]))
    print("  train 模式 batch=8 : 逐图 %.3f dB / 批MSE %.3f dB   <- 训练脚本 test_metrics.json 的口径（期望 26.654/25.39）"
          % (ps_a_train8["psnr_per_image_mean"], ps_a_train8["psnr_batch_mse"]))

    # ---------- 域 B：Nutrition5k（部署域，跨库） ----------
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if r["split"] == "test"]
    if args.limit_b:
        rows = rows[:args.limit_b]
    mean_t = torch.tensor(MEAN).view(3, 1, 1)
    std_t = torch.tensor(STD).view(3, 1, 1)
    b_imgs = []
    for row in rows:
        with Image.open(ROOT / row["image"]) as src:
            im = src.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        norm = TF.normalize(TF.to_tensor(im), MEAN, STD)
        b_imgs.append(norm * std_t + mean_t)          # 反归一化回 [0,1]，与训练口径一致
    b_rgb = torch.stack(b_imgs)
    pred_b = predict(gen, b_rgb, device)
    print("domain B (Nutrition5k test) n=%d" % len(rows))

    # ---------- 对照：把域 B 的输入按域 A 的逐图亮度/对比度对齐再生成 ----------
    def luma(t: torch.Tensor) -> torch.Tensor:
        return 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]

    lum_a, lum_b = luma(t_a), luma(b_rgb)
    m_a, s_a = float(lum_a.mean()), float(lum_a.std())
    lb = lum_b.reshape(len(lum_b), -1)
    mb = lb.mean(1).view(-1, 1, 1, 1)
    sb = lb.std(1).view(-1, 1, 1, 1).clamp_min(1e-6)
    gain = s_a / sb
    b_matched = ((b_rgb - mb) * gain + m_a).clamp(0, 1)
    pred_b_matched = predict(gen, b_matched, device)

    results = {
        "generator": str(Path(args.generator).relative_to(ROOT)),
        "domain_A": {"name": "HSIFoodIngr-64 test (generator training domain)", "n": n_a,
                     "psnr_eval_mode_batch32": ps_a,
                     "psnr_train_mode_batch8": ps_a_train8,
                     "rgb": rgb_stats(rgb_a.transpose(0, 3, 1, 2), "A rgb"),
                     "real_nir": image_stats(nir_a, "A real NIR (target)"),
                     "pred_nir": image_stats(pred_a, "A pred NIR")},
        "domain_B": {"name": "Nutrition5k frozen test (deployment domain)", "n": len(rows),
                     "rgb": rgb_stats(b_rgb.numpy(), "B rgb"),
                     "pred_nir": image_stats(pred_b, "B pred NIR"),
                     "pred_nir_contrast_matched_input": image_stats(pred_b_matched, "B pred NIR (matched)")},
        "contrast_alignment": {"A_luma_mean": m_a, "A_luma_std": s_a},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 打印对照表 ----------
    def row(d, key):
        s = d[key]
        return "%-26s %8.4f %8.4f %10.4f %10.4f %8.2f%% %9.4f" % (
            s["tag"], s["global_mean"], s["global_std"], s["per_image_contrast_avg"],
            s["per_image_span_p2_p98_avg"], 100 * s["saturated_fraction"],
            s["between_image_std_of_means"])

    print("\n== NIR 通道统计（[0,1] 域） ==")
    print("%-26s %8s %8s %10s %10s %9s %9s" % ("set", "mean", "std", "逐图对比度", "p2-p98跨度", "饱和比例", "跨图方差"))
    print(row(results["domain_A"], "real_nir"))
    print(row(results["domain_A"], "pred_nir"))
    print(row(results["domain_B"], "pred_nir"))
    print(row(results["domain_B"], "pred_nir_contrast_matched_input"))

    print("\n== RGB 输入统计 ==")
    for d in ("domain_A", "domain_B"):
        s = results[d]["rgb"]
        print("%-12s luma_mean=%.4f luma_contrast=%.4f 通道std=%s 饱和=%.2f%% 暗=%.2f%% 亮=%.2f%%" % (
            d, s["luma_mean"], s["luma_contrast_avg"],
            ["%.3f" % v for v in s["channel_std_avg"]],
            100 * s["saturated_fraction"], 100 * s["dark_fraction"], 100 * s["bright_fraction"]))

    # 差距判定
    ta = results["domain_A"]["real_nir"]
    pb = results["domain_B"]["pred_nir"]
    pm = results["domain_B"]["pred_nir_contrast_matched_input"]
    print("\n== 判定 ==")
    print("域B预测NIR 逐图对比度 = %.4f，域A实测NIR = %.4f（比值 %.2f）"
          % (pb["per_image_contrast_avg"], ta["per_image_contrast_avg"],
             pb["per_image_contrast_avg"] / max(ta["per_image_contrast_avg"], 1e-9)))
    print("域B饱和比例 = %.2f%%，域A实测 = %.2f%%" % (100 * pb["saturated_fraction"], 100 * ta["saturated_fraction"]))
    print("对比度对齐后：逐图对比度 %.4f（变化 %+.1f%%），饱和 %.2f%%"
          % (pm["per_image_contrast_avg"],
             100 * (pm["per_image_contrast_avg"] / max(pb["per_image_contrast_avg"], 1e-9) - 1),
             100 * pm["saturated_fraction"]))
    print("wrote %s" % out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
