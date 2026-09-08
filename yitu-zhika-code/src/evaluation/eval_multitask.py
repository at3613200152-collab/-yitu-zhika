"""
Phase2 Multitask Evaluation + Training Curve Plot
==================================================
1. Loads best_model.pt (Phase2 ResNet) + final_model.pth (Phase1 Generator)
2. Runs full pipeline on val split
3. Reports: MAE, MAPE, RMSE, R² for calories and weight
4. Generates scatter plot (predicted vs actual)
5. Parses phase2_run.log and plots training loss curves
"""
import os
import sys
import re
import csv
import json
import hashlib
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Environment ===
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # src/evaluation/
SRC_DIR = os.path.dirname(SCRIPT_DIR)                       # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)                     # project root
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from data.nutrition5k_loader import Nutrition5kDataset
from models.generator import UNetGenerator
from models.resnet_multitask import ResNetMultiTask
from models.checkpoint_io import load_multitask_checkpoint

# === Config ===
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Nutrition5k')
GEN_CKPT = os.path.join(PROJECT_ROOT, 'checkpoints', 'phase1', 'final_model.pth')
MODEL_CKPT = os.path.join(PROJECT_ROOT, 'checkpoints', 'multitask', 'best_model.pt')
LOG_FILE = os.path.join(PROJECT_ROOT, 'phase2_run.log')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'phase2_eval_corrected')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(os.environ.get('YITU_EVAL_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def evaluate_model():
    """Run full pipeline evaluation on val set."""
    print(f"Device: {DEVICE}")

    # 1. Val dataset
    print("\n--- Loading val dataset ---")
    verified_csv = os.path.join(DATA_DIR, 'dishes_verified.csv')
    if not os.path.exists(verified_csv):
        raise ValueError('Run verify_nutrition5k_metadata.py --write before evaluation')
    val_set = Nutrition5kDataset(root_dir=DATA_DIR, split='val', metadata_csv=verified_csv)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_set)}")

    # 2. Generator
    print("\n--- Loading Phase1 generator ---")
    generator = UNetGenerator(in_channels=3, out_channels=1, base_filters=64).to(DEVICE)
    gen_ckpt = torch.load(GEN_CKPT, map_location=DEVICE, weights_only=False)
    gen_state = gen_ckpt.get('G_state_dict', gen_ckpt.get('generator_state_dict', gen_ckpt.get('model_state_dict', gen_ckpt)))
    result = generator.load_state_dict(gen_state, strict=True)
    if result.missing_keys:
        print(f"  ⚠️ Missing keys: {len(result.missing_keys)} layers (random init)")
    if result.unexpected_keys:
        print(f"  ⚠️ Unexpected keys: {len(result.unexpected_keys)} layers (ignored)")
    if not result.missing_keys and not result.unexpected_keys:
        print(f"  ✅ All keys matched perfectly")
    generator.eval()
    print(f"Generator loaded (epoch {gen_ckpt.get('epoch', '?')})")

    # 3. ResNet model
    print("\n--- Loading Phase2 model ---")
    model, model_metadata = load_multitask_checkpoint(MODEL_CKPT, DEVICE)
    cal_index = model_metadata['target_names'].index('calories')
    mass_index = model_metadata['target_names'].index('mass')
    epoch = model_metadata.get('epoch', '?')
    print(f"Model loaded (epoch {epoch})")

    # 4. Eval loop
    print("\n--- Running evaluation ---")
    all_pred_cal, all_pred_wt = [], []
    all_gt_cal, all_gt_wt = [], []
    all_ids = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['image'].to(DEVICE)
            gt_cal = batch['calories'].numpy()
            gt_wt = batch['mass'].numpy()
            all_ids.extend(batch['dish_id'])

            # Value domain adaptation
            rgb_01 = images * STD + MEAN
            rgb_norm = rgb_01 * 2 - 1
            nir = generator(rgb_norm)
            nir_01 = (nir + 1) / 2
            nir_imagenet = (nir_01 - 0.485) / 0.229
            input_4ch = torch.cat([images, nir_imagenet], dim=1)

            outputs = model(input_4ch)
            preds = outputs['nutrition'].cpu().numpy()

            all_pred_cal.extend(preds[:, cal_index])
            all_pred_wt.extend(preds[:, mass_index])
            all_gt_cal.extend(gt_cal)
            all_gt_wt.extend(gt_wt)

            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(val_loader)}")

    all_pred_cal = np.array(all_pred_cal)
    all_pred_wt = np.array(all_pred_wt)
    all_gt_cal = np.array(all_gt_cal)
    all_gt_wt = np.array(all_gt_wt)

    # 5. Metrics
    def metrics(pred, gt, name):
        mae = np.mean(np.abs(pred - gt))
        nonzero = gt != 0
        mape = np.mean(np.abs((pred[nonzero] - gt[nonzero]) / gt[nonzero])) * 100 if nonzero.sum() > 0 else float('nan')
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f"\n  {name}:")
        print(f"    MAE:  {mae:.2f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    GT:   [{gt.min():.1f}, {gt.max():.1f}], mean={gt.mean():.1f}")
        print(f"    Pred: [{pred.min():.1f}, {pred.max():.1f}], mean={pred.mean():.1f}")
        return {'mae': mae, 'mape': mape, 'rmse': rmse, 'r2': r2}

    print("\n=== Results ===")
    cal_m = metrics(all_pred_cal, all_gt_cal, 'Calories')
    wt_m = metrics(all_pred_wt, all_gt_wt, 'Weight (g)')

    # Preserve sample-level evidence and exact input identities, not just rounded scores.
    def sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(chunk)
        return h.hexdigest()

    evidence = {
        'samples': len(all_ids), 'split': 'legacy_random_seed42_val',
        'official_benchmark': False, 'device': str(DEVICE),
        'checkpoint_target_names': model_metadata['target_names'],
        'sha256': {os.path.relpath(p, PROJECT_ROOT): sha256(p)
                   for p in (GEN_CKPT, MODEL_CKPT, verified_csv, MODEL_CKPT + '.metadata.json')},
        'calories': {k: float(v) for k, v in cal_m.items()},
        'mass': {k: float(v) for k, v in wt_m.items()},
        'mape_nonzero_calories_samples': int(np.count_nonzero(all_gt_cal)),
        'mape_nonzero_mass_samples': int(np.count_nonzero(all_gt_wt)),
        'limitations': 'Historical random validation split, not an official benchmark. '
                       'Original checkpoint has no training-data hash; output semantics reconstructed from historical code and audited CSV.',
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(evidence, f, indent=2, allow_nan=False)
    with open(os.path.join(OUTPUT_DIR, 'predictions.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dish_id', 'gt_calories_kcal', 'pred_calories_kcal', 'gt_mass_g', 'pred_mass_g'])
        writer.writerows(zip(all_ids, all_gt_cal, all_pred_cal, all_gt_wt, all_pred_wt))

    # 6. Save metrics
    metrics_file = os.path.join(OUTPUT_DIR, 'metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Phase2 Multitask Evaluation Results\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Model: best_model.pt (epoch {epoch})\n")
        f.write(f"Generator: final_model.pth (epoch 200)\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write("Corrected physical labels and explicit checkpoint target mapping.\n")
        f.write("Historical random validation split; NOT official benchmark or new-model improvement.\n\n")
        f.write("Calories:\n")
        f.write(f"  MAE:  {cal_m['mae']:.2f}\n")
        f.write(f"  MAPE: {cal_m['mape']:.2f}%\n")
        f.write(f"  RMSE: {cal_m['rmse']:.2f}\n")
        f.write(f"  R2:   {cal_m['r2']:.4f}\n\n")
        f.write("Weight:\n")
        f.write(f"  MAE:  {wt_m['mae']:.2f}\n")
        f.write(f"  MAPE: {wt_m['mape']:.2f}%\n")
        f.write(f"  RMSE: {wt_m['rmse']:.2f}\n")
        f.write(f"  R2:   {wt_m['r2']:.4f}\n")
    print(f"\nMetrics saved: {metrics_file}")

    # 7. Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pred, gt, title in [
        (axes[0], all_pred_cal, all_gt_cal, 'Calories (kcal)'),
        (axes[1], all_pred_wt, all_gt_wt, 'Weight (g)')
    ]:
        ax.scatter(gt, pred, alpha=0.4, s=20, c='steelblue', edgecolors='navy', linewidth=0.3)
        lo = min(gt.min(), pred.min())
        hi = max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
        if len(gt) > 1 and np.std(gt) > 0:
            z = np.polyfit(gt, pred, 1)
            x_fit = np.linspace(lo, hi, 100)
            ax.plot(x_fit, np.poly1d(z)(x_fit), 'g-', linewidth=1.5, alpha=0.7, label=f'Fit (slope={z[0]:.2f})')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_file = os.path.join(OUTPUT_DIR, 'scatter_plot.png')
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved: {scatter_file}")

    return cal_m, wt_m


def plot_training_curves():
    """Parse phase2_run.log and plot training loss curves."""
    print("\n--- Plotting training curves ---")
    if not os.path.exists(LOG_FILE):
        print("Log file not found, skipping curves")
        return

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    pattern = re.compile(r'Epoch \[(\d+)/\d+\] Train_loss=([\d.]+) Train_acc=([\d.]+) Val_loss=([\d.]+) Val_acc=([\d.]+)')

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                train_losses.append(float(m.group(2)))
                train_accs.append(float(m.group(3)))
                val_losses.append(float(m.group(4)))
                val_accs.append(float(m.group(5)))

    if not train_losses:
        print("No epoch data found in log")
        return

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=1.5, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Phase2 Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val) + 1
    ax1.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.5)
    ax1.annotate(f'Best: {best_val:.2f} (ep{best_epoch})', xy=(best_epoch, best_val),
                xytext=(best_epoch + 5, best_val + 5), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green'))

    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', linewidth=1.5, label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', linewidth=1.5, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Phase2 Classification Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.99, 1.01)

    plt.tight_layout()
    curve_file = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(curve_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {curve_file}")
    print(f"  Epochs parsed: {len(train_losses)}")
    print(f"  Best val loss: {best_val:.2f} (epoch {best_epoch})")
    print(f"  Final train loss: {train_losses[-1]:.2f}")
    print(f"  Final val loss: {val_losses[-1]:.2f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase2 Evaluation + Training Curves")
    print("=" * 60)

    cal_m, wt_m = evaluate_model()
    plot_training_curves()

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
