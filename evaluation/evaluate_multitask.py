"""
阶段二评估脚本: 多任务营养估计网络
====================================
在test集上评估ResNet50多任务网络(分类+营养回归)。

指标:
    分类: Accuracy, Macro-F1(仅统计有样本的类), 每类P/R/F1表, 混淆矩阵
    回归: kcal MAE/RMSE/MAPE, 质量(g) MAE/RMSE
    回归预测为log1p空间, 评估前expm1还原到物理空间。

使用方法:
    & .venv\\Scripts\\python.exe evaluation\\evaluate_multitask.py --generator_ckpt checkpoints\\generator\\best_model.pt
    & .venv\\Scripts\\python.exe evaluation\\evaluate_multitask.py --split val --generator_ckpt checkpoints\\generator\\best_model.pt
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.multitask.resnet_multitask import ResNetMultiTask
from models.generator.unet import UNetGenerator
from data.nutrition5k_loader import create_nutrition5k_dataloader, CATEGORY_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="阶段二: 评估多任务营养估计网络")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/multitask/best_model.pt",
                        help="多任务网络checkpoint路径")
    parser.add_argument("--generator_ckpt", type=str, default=None,
                        help="阶段一生成器checkpoint路径(不传则NIR置零)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="评估的数据划分 (默认test)")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_generator(generator_ckpt: str, device) -> UNetGenerator:
    """加载阶段一NIR生成器, 输出[-1,1]域"""
    generator = UNetGenerator(input_channels=3, output_channels=1).to(device)
    ckpt = torch.load(generator_ckpt, map_location=device)
    gen_state = ckpt.get('model_state_dict', ckpt)
    gen_state = {k.replace('generator.', ''): v
                 for k, v in gen_state.items() if k.startswith('generator.')}
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
    loader = create_nutrition5k_dataloader(
        root_dir=data_path, split=args.split,
        img_size=data_cfg.get('img_size', 256),
        batch_size=mt_cfg.get('batch_size', 32),
        num_workers=data_cfg.get('num_workers', 4),
        augmentation=False,
    )
    num_classes = len(CATEGORY_CLASSES)
    print(f"{args.split}集: {len(loader.dataset)} 样本, {num_classes} 类(含空类)")

    # ======== 模型 ========
    model = ResNetMultiTask(
        num_classes=mt_cfg.get('num_classes', num_classes),
        input_channels=mt_cfg.get('input_channels', 4),
        pretrained=False,
    ).to(device)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"[错误] checkpoint不存在: {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"已加载: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

    # ======== 阶一生成器(可选) ========
    generator = None
    if args.generator_ckpt and os.path.exists(args.generator_ckpt):
        generator = load_generator(args.generator_ckpt, device)
        print(f"NIR生成器已加载: {args.generator_ckpt}")
    else:
        print("未加载NIR生成器, NIR通道置零(仅限对照实验)")

    # ImageNet反归一化常量(与train_multitask.py保持一致)
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

            # log1p空间 → expm1还原物理空间
            cal_pred_list.append(torch.expm1(outputs['nutrition'][:, 0]).clamp(min=0.0).cpu())
            mass_pred_list.append(torch.expm1(outputs['nutrition'][:, 1]).clamp(min=0.0).cpu())
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
        cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    pred_cnt = cm.sum(0).astype(np.float64)
    true_cnt = cm.sum(1).astype(np.float64)
    present = true_cnt > 0
    precision = np.where(pred_cnt > 0, tp / np.maximum(pred_cnt, 1), 0.0)
    recall = np.where(true_cnt > 0, tp / np.maximum(true_cnt, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    macro_f1 = float(f1[present].mean())
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
    out_dir = os.path.join(config.get('output_dir', './output'), "multitask_eval")
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append(f"多任务营养估计网络评估 split={args.split}")
    lines.append(f"checkpoint={ckpt_path}")
    lines.append(f"NIR生成器={'已加载' if generator is not None else '未加载(NIR置零)'}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"样本数: {n}")
    lines.append(f"类别数: {num_classes} (其中有样本的类: {n_present}, 空类: "
                 f"{num_classes - n_present} → {', '.join(CATEGORY_CLASSES[i] for i in range(num_classes) if not present[i]) or '无'})")
    lines.append("")
    lines.append("[分类指标]")
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append(f"Macro-F1 (仅统计有样本的{n_present}类): {macro_f1:.4f}")
    lines.append("")
    lines.append("[回归指标] (log1p预测已expm1还原到物理空间)")
    lines.append(f"卡路里: MAE={cal_mae:.1f} kcal, RMSE={cal_rmse:.1f} kcal, MAPE={cal_mape:.1f}%")
    lines.append(f"质量:   MAE={mass_mae:.1f} g, RMSE={mass_rmse:.1f} g")
    lines.append("")
    lines.append("[每类 P/R/F1/support]")
    lines.append(f"{'类别':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for i in range(num_classes):
        lines.append(f"{CATEGORY_CLASSES[i]:<16}{precision[i]:>10.4f}{recall[i]:>10.4f}"
                     f"{f1[i]:>10.4f}{int(true_cnt[i]):>10d}")
    lines.append("")
    lines.append("[混淆矩阵] (行=真值, 列=预测)")
    header = "true\\pred".ljust(16) + "".join(CATEGORY_CLASSES[j][:6].rjust(8) for j in range(num_classes))
    lines.append(header)
    for i in range(num_classes):
        row = CATEGORY_CLASSES[i][:14].ljust(16) + "".join(str(int(cm[i, j])).rjust(8) for j in range(num_classes))
        lines.append(row)

    summary = "\n".join(lines)
    print(summary)

    with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)
    print(f"\n汇总已保存: {os.path.join(out_dir, 'eval_summary.txt')}")

    # 混淆矩阵图(可选, matplotlib可用时保存)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(CATEGORY_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CATEGORY_CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (split={args.split}, acc={acc:.3f}, macroF1={macro_f1:.3f})")
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > 0:
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=7)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        print(f"混淆矩阵图已保存: {os.path.join(out_dir, 'confusion_matrix.png')}")
    except Exception as e:
        print(f"(matplotlib不可用, 跳过混淆矩阵图: {e})")


if __name__ == "__main__":
    main()
