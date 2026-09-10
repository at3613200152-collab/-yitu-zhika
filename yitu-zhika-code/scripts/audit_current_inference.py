"""Read-only CPU diagnostic; never trains, downloads, serves or replaces weights.

Writes a new evidence JSON only when --output is specified; existing output is
not overwritten. File names describe probe images, not verified ground truth.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Choose a new evidence filename; do not overwrite prior audits")
    manifest_path = ROOT / "results/meal_official_v1/manifest.json"
    manifest = read(manifest_path)
    names = {i: name for name, i in manifest["category_to_idx"].items()}
    evidence = {
        "scope": "local CPU source/artifact diagnostic, not a fresh benchmark or training run",
        "manifest_sha256": digest(manifest_path),
        "target_names": manifest["target_names"],
        "category_to_idx": manifest["category_to_idx"],
        "category_source": manifest["category_source"],
        "category_counts": manifest["category_counts"],
        "models": {},
        "probe_ground_truth": "Unverified; image filename must not be treated as a class/weight/nutrition label."
    }
    expanded_path = ROOT / "results/meal_expanded_v2/manifest.json"
    if expanded_path.exists():
        expanded = read(expanded_path)
        old = {r["dish_id"]: r for r in manifest["rows"]}
        additions = [r for r in expanded["rows"] if r["dish_id"] not in old]
        evidence["expanded"] = {
            "sha256": digest(expanded_path),
            "new_dishes": len(additions),
            "new_splits": dict(Counter(r["split"] for r in additions)),
            "counts": dict(Counter(r["split"] for r in expanded["rows"])),
            "current_models_use_expanded": False
        }

    for name in ["meal_rgb_official_v1", "meal_nir_official_v1"]:
        folder = ROOT / "results" / name
        metric = read(folder / "test_metrics.json")
        protocol = read(folder / "protocol.json")
        with (folder / "test_predictions.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        true = [int(r["true_coarse_class"]) for r in rows]
        pred = [int(r["pred_coarse_class"]) for r in rows]
        classes = []
        for i, label in sorted(names.items()):
            support = true.count(i)
            predicted = pred.count(i)
            tp = sum(t == i and p == i for t, p in zip(true, pred))
            precision = tp / predicted if predicted else 0
            recall = tp / support if support else None
            f1 = 2 * tp / (support + predicted) if support + predicted else 0
            classes.append({"name": label, "support": support, "predicted": predicted,
                            "precision": precision, "recall": recall, "f1": f1})
        checkpoint = ROOT / "checkpoints" / name / "best.pt"
        sha = digest(checkpoint)
        evidence["models"][name] = {
            "checkpoint_sha256": sha,
            "hash_matches_metrics": sha == metric["checkpoint_sha256"],
            "target_names": protocol.get("target_names"),
            "manifest_sha256": metric["manifest_sha256"],
            "saved_metrics": metric["metrics"],
            "per_class": classes,
            "macro_f1_supported": sum(c["f1"] for c in classes if c["support"]) / sum(bool(c["support"]) for c in classes),
            "balanced_accuracy_supported": sum(c["recall"] for c in classes if c["support"]) / sum(bool(c["support"]) for c in classes),
            "pred_distribution": {names[i]: pred.count(i) for i in names},
            "probe_results": []
        }

    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF
    from src.training.train_meal_official import MealNet, MEAN, STD
    from src.training.train_meal_nir_official import NirMealNet
    torch.set_num_threads(2)
    torch.manual_seed(42)
    probes = [p for p in (ROOT.parent / "test_images").iterdir()
              if p.name in {"test_apple_pie.jpg", "test_baby_back_ribs.jpg", "test_bibimbap.jpg",
                            "test_bread_pudding.jpg", "test_caesar_salad.jpg", "user_mixed_meal.png"}]
    for name, item in evidence["models"].items():
        checkpoint = torch.load(ROOT / "checkpoints" / name / "best.pt", map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        item["head_shapes"] = {k: list(v.shape) for k, v in state.items()
                               if k in ("network.classifier.1.weight", "network.regressor.3.weight", "network.features.0.weight")}
        item["checkpoint_epoch"] = checkpoint["epoch"]
        item["checkpoint_matches_protocol"] = checkpoint["config"] == read(ROOT / "results" / name / "protocol.json")
        if name == "meal_rgb_official_v1":
            model = MealNet(manifest, pretrained=False)
        else:
            generator = {k.removeprefix("generator."): v for k, v in state.items() if k.startswith("generator.")}
            model = NirMealNet(manifest, generator, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            for path in sorted(probes):
                with Image.open(path) as source:
                    image = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                tensor = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0)
                logits, values = model(tensor)
                probabilities = logits.softmax(1)[0]
                top = probabilities.topk(3)
                probe = {
                    "file": path.name, "sha256": digest(path),
                    "top3": [{"category": names[int(i)], "score": float(p)} for p, i in zip(top.values, top.indices)],
                    "calories": float(values[0, 0]), "mass": float(values[0, 1]),
                    "finite": bool(torch.isfinite(logits).all() and torch.isfinite(values).all())
                }
                item["probe_results"].append(probe)
                print(json.dumps({"model": name, **probe}, ensure_ascii=False), flush=True)
        del model, checkpoint, state
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": {k: {q: v[q] for q in ("hash_matches_metrics", "head_shapes",
          "macro_f1_supported", "balanced_accuracy_supported", "pred_distribution")} for k, v in evidence["models"].items()},
          "category_counts": evidence["category_counts"], "expanded": evidence.get("expanded")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
