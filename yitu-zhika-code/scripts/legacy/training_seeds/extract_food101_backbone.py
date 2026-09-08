"""Extract the ResNet50 backbone (drop 101-class head) from a Food-101 best.pt."""
import json
from pathlib import Path

import torch

CKPT = Path(r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\checkpoints\food101_pretrained\best.pt")
OUT = CKPT.parent / "backbone_only.pt"

ck = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = ck["model"] if "model" in ck else ck
backbone = {k: v for k, v in sd.items() if not k.startswith("fc.")}
out = {"model": backbone,
       "best_epoch": ck.get("best_epoch"),
       "best_test_acc": ck.get("best_acc"),
       "source": "pretrain_food101.py (stopped at epoch 4, 76.0% test_acc)",
       "note": "Stripped 101-class fc; only ResNet50 conv body + bn + downsample."}
torch.save(out, OUT)
print(f"Saved {OUT} ({len(backbone)} tensors)")
print(f"Best epoch: {ck.get('best_epoch')}, best test_acc: {ck.get('best_acc')}")
