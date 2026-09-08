"""TTA (test-time augmentation) evaluation for a trained meal_seed checkpoint.

Loads results/meal_<tag>_seed<seed>/best.pt and re-evaluates on the
frozen Nutrition5k test split using horizontal-flip + multi-scale averaging.

Outputs:
  results/meal_<tag>_seed<seed>/tta_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

# ROOT is fixed to the project tree; do not derive from __file__.
ROOT = Path(r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code")
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest
from src.training.train_meal_seed import (
    MANIFEST, MANIFEST_SHA, MEAN, STD, MealDataset, MealNet, regression_metrics,
)


def predict_with_tta(model, image_pil, sizes=(224, 256, 288), device='cuda'):
    preds, logits_acc = [], []
    for sz in sizes:
        for flip in (False, True):
            img = image_pil.convert('RGB').resize((sz, sz), Image.Resampling.BILINEAR)
            if flip:
                img = ImageOps.mirror(img)
            x = TF.normalize(TF.to_tensor(img), MEAN, STD).unsqueeze(0).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, pred = model(x)
            preds.append(pred.float().cpu().numpy()[0])
            logits_acc.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
    pred_mean = np.mean(preds, axis=0)
    logits_mean = np.mean(logits_acc, axis=0)
    return pred_mean, logits_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()

    OUTPUT = ROOT / 'results' / f'meal_{args.tag}_seed{args.seed}'
    WEIGHTS = ROOT / 'checkpoints' / f'meal_{args.tag}_seed{args.seed}'
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if file_digest(MANIFEST) != MANIFEST_SHA:
        raise SystemExit('Manifest hash mismatch')

    ck = torch.load(WEIGHTS / 'best.pt', map_location='cpu', weights_only=True)
    model = MealNet(manifest, pretrained=False)
    model.load_state_dict(ck['model'])
    model.cuda().eval()
    best_epoch = ck['best_epoch']

    rows = [r for r in manifest['rows'] if r['split'] == 'test']
    gts, preds, cids, cpreds, ids = [], [], [], [], []
    for row in rows:
        with Image.open(ROOT / row['image']) as img:
            pred, logits = predict_with_tta(model, img)
        preds.append(pred)
        gts.append(row['targets'])
        cids.append(row['category_idx'])
        cpreds.append(int(logits.argmax()))
        ids.append(row['dish_id'])

    preds = np.array(preds); gts = np.array(gts, dtype=np.float64)
    cls_acc = float(np.mean(np.array(cpreds) == np.array(cids)))
    metrics = {**regression_metrics(gts, preds), 'class_accuracy': cls_acc,
               'best_epoch': best_epoch, 'tta_passes': 6}
    out = OUTPUT / 'tta_metrics.json'
    atomic_json(out, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
