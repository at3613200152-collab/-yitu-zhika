"""Phase2 多任务网络完整评估 (吸收自队友 pure-pix2pix 分支)

来源: 队友 `evaluation/evaluate_multitask.py`
适配: - import 路径改为 src/ 风格 (from models.xxx → sys.path.insert(src))
      - 11 类 (我们 v2 配置) 替代队友 16 类
      - 5 维回归 (calories/mass/protein/carb/fat) 替代队友 2 维 + log1p
      - 直接回归，无需 expm1 还原
      - 类别列表从 dataset 实例动态获取 (我们没有全局 CATEGORY_CLASSES)

新增指标 (相比原 eval_multitask.py):
    分类: Accuracy, Macro-F1, 每类 P/R/F1, 混淆矩阵
    回归: kcal/mass MAE/RMSE/MAPE (原脚本只有这些)
    输出: eval_summary.txt + confusion_matrix.npy

用法:
    python src/evaluation/eval_multitask_full.py \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/meal_rgb_official_v1/best.pt \\
        [--generator_ckpt checkpoints/hsi_unet_v2/best.pt] \\
        [--split val]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

# === 路径适配 (我们项目用 src/ 子目录) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)

from data.nutrition5k_loader import Nutrition5kDataset, create_nutrition5k_dataloader
from models.generator import UNetGenerator
from models.resnet_multitask import ResNetMultiTask
from models.checkpoint_io import load_multitask_checkpoint
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Phase2 多任务网络完整评估 (含分类指标 + 混淆矩阵)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/meal_rgb_official_v1/best.pt",
                        help="多任务网络 checkpoint 路径")
    parser.add_argument("--generator_ckpt", type=str, default=None,
                        help="阶段一生成器 checkpoint 路径 (不传则 NIR 置零)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="评估的数据划分 (默认 val)")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_generator(generator_ckpt: str, device) -> UNetGenerator:
    """加载阶段一 NIR 生成器，输出 [-1, 1] 域"""
    generator = UNetGenerator(input_channels=3, output_channels=1).to(device)
    ckpt = torch.load(generator_ckpt, map_location=device, weights_only=False)
    gen_state = ckpt.get('model_state_dict', ckpt)
    # 兼容 Pix2Pix 训练 checkpoint 的 generator. 前缀
    if any(k.startswith('generator.') for k in gen_state):
        gen_state = {k.replace('generator.', '', 1): v
                     for k, v in gen_state.items() if k.startswith('generator.')}
    else:
        gen_state = {k.replace('module.', ''): v
                     for k, v in gen_state.items() if 'discriminator' not in k}
    generator.load_state_dict(gen_state, strict=False)
    generator.eval()
    return generator


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    mt_cfg = config.get('multitask', {})
    data_cfg = config.get('data', {})

    device_str = args.device or config.get('device', 'cuda')
    if not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"设备: {device}")

    # ======== 数据 ========
    data_path = data_cfg.get('nutrition5k_path', './data/Nutrition5k')
    dataset = Nutrition5kDataset(
        root_dir=data_path,
        split=args.split,
        img_size=data_cfg.get('img_size', 256),
        augmentation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=mt_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 2),
        pin_memory=True,
    )

    # 从 dataset 实例获取类别列表（我们没有全局 CATEGORY_CLASSES 常量）
    category_names = sorted(dataset.category_to_idx.keys(), key=lambda c: dataset.category_to_idx[c])
    num_classes = len(category_names)
    print(f"{args.split} 集: {len(dataset)} 样本, {num_classes} 类")

    # ======== 模型 ========
    model = ResNetMultiTask(
        num_classes=mt_cfg.get('num_classes', num_classes),
        input_channels=mt_cfg.get('input_channels', 4),
        pretrained=False,
    ).to(device)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"[错误] checkpoint 不存在: {ckpt_path}")
        sys.exit(1)
    # 用我们项目的 load_multitask_checkpoint 兼容加载
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    except Exception as e:
        # 降级到 load_multitask_checkpoint
        print(f"直接加载失败 ({e})，尝试 load_multitask_checkpoint...")
        load_multitask_checkpoint(model, ckpt_path, device=device)
    model.eval()
    print(f"已加载: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

    # ======== 阶一生成器 (可选) ========
    generator = None
    if args.generator_ckpt and os.path.exists(args.generator_ckpt):
        generator = load_generator(args.generator_ckpt, device)
        print(f"NIR 生成器已加载: {args.generator_ckpt}")
    else:
        print("未加载 NIR 生成器, NIR 通道置零 (仅限对照实验)")

    # ImageNet 反归一化常量
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def make_4ch(images_tensor: torch.Tensor) -> torch.Tensor:
        if generator is not None:
            with torch.no_grad():
                rgb01 = images_tensor * imagenet_std + imagenet_mean
                nir = generator(rgb01 * 2.0 - 1.0)
            return torch.cat([images_tensor, nir], dim=1)
        nir_zero = torch.zeros(
            images_tensor.shape[0], 1, images_tensor.shape[2], images_tensor.shape[3],
            device=images_tensor.device,
        )
        return torch.cat([images_tensor, nir_zero], dim=1)

    # ======== 推理 ========
    print("评估中...")
    all_logits, all_labels = [], []
    cal_pred_list, cal_gt_list = [], []
    mass_pred_list, mass_gt_list = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['category_idx'].to(device)
            input_4ch = make_4ch(images)
            outputs = model(input_4ch)

            all_logits.append(outputs['logits'].cpu())
            all_labels.append(labels.cpu())

            # 我们 v2 是直接回归 (不取 log)，无需 expm1 还原
            nutrition = outputs['nutrition'].cpu()
            cal_pred_list.append(nutrition[:, 0].clamp(min=0.0))
            mass_pred_list.append(nutrition[:, 1].clamp(min=0.0))
            cal_gt_list.append(batch['calories'].float().cpu())
            mass_gt_list.append(batch['mass'].float().cpu())

    logits = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    cal_pred = torch.cat(cal_pred_list).numpy()
    cal_gt = torch.cat(cal_gt_list).numpy()
    mass_pred = torch.cat(mass_pred_list).numpy()
    mass_gt = torch.cat(mass_gt_list).numpy()

    preds = logits.argmax(1)
    n = len(labels_np)

    # ======== 分类指标 ========
    acc = float((preds == labels_np).mean())

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=true, cols=pred
    for t, p in zip(labels_np, preds):
        # 防止索引越界 (类别索引可能超过 num_classes)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    pred_cnt = cm.sum(0).astype(np.float64)
    true_cnt = cm.sum(1).astype(np.float64)
    present = true_cnt > 0
    precision = np.where(pred_cnt > 0, tp / np.maximum(pred_cnt, 1), 0.0)
    recall = np.where(true_cnt > 0, tp / np.maximum(true_cnt, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    n_present = int(present.sum())

    # ======== 回归指标 ========
    cal_err = cal_pred - cal_gt
    cal_mae = float(np.abs(cal_err).mean())
    cal_rmse = float(np.sqrt((cal_err ** 2).mean()))
    cal_mape = float((np.abs(cal_err) / np.maximum(cal_gt, 1e-6)).mean() * 100)

    mass_err = mass_pred - mass_gt
    mass_mae = float(np.abs(mass_err).mean())
    mass_rmse = float(np.sqrt((mass_err ** 2).mean()))

    # ======== 输出 ========
    out_dir = os.path.join(config.get('output_dir', './output'), "multitask_eval_full")
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append(f"多任务营养估计网络评估 (full) split={args.split}")
    lines.append(f"checkpoint={ckpt_path}")
    lines.append(f"NIR 生成器={'已加载' if generator is not None else '未加载(NIR置零)'}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"样本数: {n}")
    lines.append(f"类别数: {num_classes} (其中有样本的类: {n_present}, 空类: "
                 f"{num_classes - n_present})")
    lines.append("")
    lines.append("[分类指标]")
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append(f"Macro-F1 (仅统计有样本的 {n_present} 类): {macro_f1:.4f}")
    lines.append("")
    lines.append("[回归指标] (直接回归空间，无 log1p/expm1 转换)")
    lines.append(f"卡路里: MAE={cal_mae:.1f} kcal, RMSE={cal_rmse:.1f} kcal, MAPE={cal_mape:.1f}%")
    lines.append(f"质量:   MAE={mass_mae:.1f} g, RMSE={mass_rmse:.1f} g")
    lines.append("")
    lines.append("[每类 P/R/F1/support]")
    lines.append(f"{'类别':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for i in range(num_classes):
        name = (category_names[i][:14] if i < len(category_names) else f"cls_{i}")
        lines.append(f"{name:<16}{precision[i]:>10.4f}{recall[i]:>10.4f}"
                     f"{f1[i]:>10.4f}{int(true_cnt[i]):>10d}")
    lines.append("")
    lines.append("[混淆矩阵] (行=真值, 列=预测)")
    header = "true\\pred".ljust(16) + "".join(
        (category_names[j][:6] if j < len(category_names) else f"c{j}").rjust(8)
        for j in range(num_classes)
    )
    lines.append(header)
    for i in range(num_classes):
        name = category_names[i][:14] if i < len(category_names) else f"cls_{i}"
        row = name.ljust(16) + "".join(str(int(cm[i, j])).rjust(8) for j in range(num_classes))
        lines.append(row)

    summary = "\n".join(lines)
    print(summary)

    with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)
    print(f"\n输出已保存到: {out_dir}")


if __name__ == "__main__":
    main()
