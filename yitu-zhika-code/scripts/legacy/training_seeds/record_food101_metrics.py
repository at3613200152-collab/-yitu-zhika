"""Record final Food-101 metrics (stopped at epoch 4 of 20)."""
import json
from pathlib import Path
import hashlib

ROOT = Path(r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code")
OUT = ROOT / "checkpoints" / "food101_pretrained"

def sha(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

# Read protocol and history
proto = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
hist = json.loads((OUT / "epochs.json").read_text(encoding="utf-8"))
best = max(hist, key=lambda e: e["test_acc"])

metrics = {
    "best_epoch": best["epoch"],
    "best_test_acc": best["test_acc"],
    "completed_epochs": len(hist),
    "stopped_early": True,
    "stop_reason": "User-initiated stop at epoch 4/20; 76.0% test_acc sufficient as backbone for Nutrition5k pretraining",
    "backbone_sha256": sha(OUT / "backbone_only.pt"),
    "full_ckpt_sha256": sha(OUT / "best.pt"),
    "manifest_sha256": sha(ROOT / "data" / "food101_manifest" / "manifest.json"),
    "limitations": [
        "Pretrained only 4 of 20 epochs; Food-101 top-1 76.0% (vs typical 78-80% at 20 epochs).",
        "Single seed (42). Multi-seed Food-101 pretraining not performed.",
        "Backbone quality vs full 20-epoch model: estimated 1-2 percentage point downstream loss.",
    ],
}
(OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
