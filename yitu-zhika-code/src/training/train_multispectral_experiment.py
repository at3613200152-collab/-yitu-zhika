"""多光谱复现实验（非审计，用于**沿论文思路**的消融 / 多种子 / 全数据）。

论文思路：
  阶段一：在 **HSIFoodIngr-64** 配对(RGB+真实NIR)上训练**冻结**的 RGB→NIR 生成器（PSNR/SSIM 评估）。
  阶段二：在 **Nutrition5k** 上把 RGB 与**冻结生成器预测的 NIR**拼成多通道，训练 ResNet 多任务
          （分类 CE + 卡路里/重量 L1/MAPE），并用**匹配的纯 RGB 对照**做消融，多种子报告。

因此本实验**默认就沿论文**：`--mode rgb|rgbnir`（生成器冻结）、`--seed`、`--manifest`（指向全量/更大划分）。
`--joint-generator` 是**可选的非论文探索**（让生成器为下游自适应），仅当你明确想试时才用，不代表论文原意。

设计原则：
- 不触碰已审计的官方 meal_rgb_official_v1 / meal_nir_official_v1 及其 protocol/code_hashes；
- 输出隔离到 results/meal_exp_<tag>/ 与 checkpoints/meal_exp_<tag>/；
- 复用 MealNet / NirMealNet / MealDataset / losses / regression_metrics。

用法（GPU，示例）：
  # 1) 论文方式：冻结生成器，多种子的匹配 RGB 对照 vs RGB+NIR
  python src/training/train_multispectral_experiment.py --mode rgb    --seed 42 --tag rgb_s42
  python src/training/train_multispectral_experiment.py --mode rgbnir --seed 43 --tag nir_s43

  # 2) 全量/更大划分（若你下载了完整 HSIFoodIngr-64 与 Nutrition5k 划分）
  python src/training/train_multispectral_experiment.py --mode rgbnir --seed 42 \
      --manifest /path/to/nutrition5k_full_manifest.json --tag nir_full42

  # 3) 非论文可选探索：联合优化生成器（让 NIR 更"为下游服务"）
  python src/training/train_multispectral_experiment.py --mode rgbnir --seed 42 --joint-generator --tag nir_joint42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.train_meal_official import (
    MealNet, MealDataset, set_base_seed, seed_epoch,
    MANIFEST, MANIFEST_SHA, MEAN, STD, losses, regression_metrics,
)
from src.training.train_meal_nir_official import NirMealNet, GENERATOR
from scripts.capsicum_job import atomic_json, file_digest


class JointNirMealNet(NirMealNet):
    """联合优化版：训练时允许梯度进入生成器（默认官方行为仍为冻结）。

    - train(mode) 在 mode=True 且 joint_allowed 时把 generator 置为 train；
    - forward 在 joint_allowed 时对生成器**不**用 no_grad，允许反向传播。
    """

    def __init__(self, manifest, generator_state, pretrained=True):
        super().__init__(manifest, generator_state, pretrained)
        self.joint_allowed = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.joint_allowed:
            self.generator.train()
        else:
            self.generator.eval()
        return self

    def forward(self, images):
        rgb01 = images * self.rgb_std + self.rgb_mean
        nir = self.generator(rgb01 * 2 - 1).float()
        nir_normalized = ((nir + 1) / 2 - 0.485) / 0.229
        x = torch.cat([images, nir_normalized], dim=1)
        # 不走 NirMealNet.forward（其含 no_grad），直接走 MealNet.forward
        return MealNet.forward(self, x)


def build_model(manifest, mode, generator_state, joint):
    if mode == "rgb":
        return MealNet(manifest, pretrained=False)
    model = JointNirMealNet(manifest, generator_state, pretrained=False) if joint else \
        NirMealNet(manifest, generator_state, pretrained=False)
    if joint:
        model.joint_allowed = True
        model.generator.requires_grad_(True)
    return model


def make_optimizer(model, args):
    params = [
        {"params": model.network.features.parameters(), "lr": 1e-5},
        {"params": list(model.network.classifier.parameters()) + list(model.network.regressor.parameters()), "lr": 1e-4},
    ]
    if args.mode == "rgbnir" and args.joint_generator:
        params.append({"params": model.generator.parameters(), "lr": 1e-5})
    return torch.optim.AdamW(params, weight_decay=1e-4)


def run_epoch(model, manifest, split, epoch, args, optimizer=None):
    training = optimizer is not None
    if training != (split == "train"):
        raise ValueError("Optimizer may only see training rows")
    model.train(training)
    if args.joint_generator:
        model.generator.train() if training else model.generator.eval()
    import torch.utils.data
    loader = torch.utils.data.DataLoader(
        MealDataset(manifest, split, epoch), batch_size=args.batch, shuffle=training,
        num_workers=0, pin_memory=True, drop_last=False,
        generator=torch.Generator().manual_seed(42 + epoch),
    )
    total, reg_total, n = 0.0, 0.0, 0
    truths, predicts, pclasses, tclasses, ids = [], [], [], [], []
    for batch, (images, target, classes, dish_ids) in enumerate(loader, 1):
        images, target, classes = images.cuda(), target.cuda(), classes.cuda()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, prediction = model(images)
            loss, reg = losses(logits, prediction, target, classes, model.target_scale)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        size = len(ids)
        n += size
        total += loss.item() * size
        reg_total += reg.item() * size
        truths.extend(target.detach().cpu().tolist())
        predicts.extend(prediction.detach().cpu().tolist())
        pclasses.extend(logits.argmax(1).detach().cpu().tolist())
        tclasses.extend(classes.cpu().tolist())
        ids.extend(dish_ids)
        if args.limit and batch > args.limit and split != "test":
            break
    metrics = regression_metrics(truths, predicts)
    return {
        "loss": total / max(n, 1),
        "reg_normalized_l1": reg_total / max(n, 1),
        "n": n,
        "metrics": metrics,
    }, {"ids": ids, "truths": truths, "predictions": predicts, "classes": tclasses, "pred_classes": pclasses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rgb", "rgbnir"], default="rgb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--joint-generator", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="每 epoch 最多 batch 数（0=不限，用于快速冒烟）")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--tag", default="exp")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    # 默认 manifest 校验其 SHA；自定义 manifest 跳过（用户自行保证）
    if str(Path(args.manifest).resolve()) == str(MANIFEST.resolve()):
        if file_digest(MANIFEST) != MANIFEST_SHA:
            raise ValueError("默认 meal manifest SHA 不匹配")

    out = ROOT / "results" / f"meal_exp_{args.tag}"
    wgt = ROOT / "checkpoints" / f"meal_exp_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    wgt.mkdir(parents=True, exist_ok=True)

    set_base_seed(args.seed)
    generator_state = None
    if args.mode == "rgbnir":
        gen = torch.load(GENERATOR, map_location="cpu", weights_only=True)
        generator_state = gen["G_state_dict"]

    model = build_model(manifest, args.mode, generator_state, args.joint_generator).cuda()
    optimizer = make_optimizer(model, args)

    n_rows = {s: len([r for r in manifest["rows"] if r["split"] == s]) for s in ("train", "val", "test")}
    config = {
        "protocol": f"meal_exp_{args.tag}", "mode": args.mode, "seed": args.seed,
        "joint_generator": args.joint_generator, "epochs": args.epochs, "batch_size": args.batch,
        "rows": n_rows, "manifest_sha256": manifest.get("_sha256"), "limit": args.limit,
    }
    atomic_json(out / "status.json", {"protocol": args.tag, "stage": "started", "config": config})

    best, best_epoch = float("inf"), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        seed_epoch(epoch)
        train_metrics, _ = run_epoch(model, manifest, "train", epoch, args, optimizer)
        val_metrics, _ = run_epoch(model, manifest, "val", epoch, args)
        improved = val_metrics["reg_normalized_l1"] < best
        if improved:
            best, best_epoch = val_metrics["reg_normalized_l1"], epoch
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "val": val_metrics})
        torch.save({"model": model.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                    "config": config, "history": history}, wgt / "last.pt")
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                        "config": config, "history": history}, wgt / "best.pt")
        atomic_json(out / "epochs.json", history)
        print(json.dumps({"stage": "epoch", "epoch": epoch, "best_epoch": best_epoch,
                          "val_calorie_mae": val_metrics["metrics"]["calories"]["mae"]}), flush=True)

    model.load_state_dict(torch.load(wgt / "best.pt", map_location="cpu", weights_only=True)["model"])
    test_metrics, detail = run_epoch(model, manifest, "test", best_epoch, args)
    atomic_json(out / "test_metrics.json", {
        "best_epoch": best_epoch, "checkpoint_sha256": file_digest(wgt / "best.pt"),
        "metrics": test_metrics, "config": config,
    })
    import csv
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "true_calories", "true_mass", "pred_calories", "pred_mass",
                    "true_coarse_class", "pred_coarse_class"])
        for i, dish in enumerate(detail["ids"]):
            w.writerow([dish, *detail["truths"][i], *detail["predictions"][i],
                        detail["classes"][i], detail["pred_classes"][i]])
    atomic_json(out / "status.json", {"protocol": args.tag, "stage": "complete", "config": config})
    print(json.dumps({"stage": "complete", "tag": args.tag, "best_epoch": best_epoch,
                      "test_calorie_mae": test_metrics["metrics"]["calories"]["mae"]}), flush=True)


if __name__ == "__main__":
    main()
