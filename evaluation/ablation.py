"""
消融实验脚本 (真实数据版)
==========================
课设核心论点: 预测NIR通道对食物营养估计的贡献。

在相同数据划分(默认test)上对比两组已训练模型:
    1. 主实验  RGB+预测NIR : 阶段一生成器输出的NIR拼接到RGB, 送入多任务网络
    2. 对照组  RGB-only    : NIR通道置零, 其余超参/增强/种子与主实验完全一致

评估通过调用 evaluation/evaluate_multitask.py 完成, 指标口径与单独评估完全一致。
本脚本负责:
    - 依次运行两组评估并实时转发输出
    - 解析关键指标, 生成对比报告(文本 + JSON)
    - 分组归档 eval_summary.txt / confusion_matrix.png (避免两组互相覆盖)

说明: 旧版脚本的7组消融设想(高级生成器/多波段/注意力/烹饪头/分割)
      未在本课设实现(数据集仅含单一NIR波段), 相关占位框架已移除。

用法:
    cd D:\\-yitu-zhika
    & .venv\\Scripts\\python.exe evaluation\\ablation.py
"""

import os
import sys
import re
import json
import shutil
import argparse
import subprocess
from datetime import datetime

# 项目根目录 (本脚本位于 <root>/evaluation/ 下)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "evaluation", "evaluate_multitask.py")
DEFAULT_EVAL_OUTPUT = os.path.join(PROJECT_ROOT, "output", "multitask_eval")
EVAL_ARTIFACTS = ("eval_summary.txt", "confusion_matrix.png")


def parse_args():
    parser = argparse.ArgumentParser(description="消融实验: RGB+预测NIR vs RGB-only")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--main_ckpt", type=str, default="./checkpoints/multitask/best_model.pt",
                        help="主实验checkpoint (RGB+预测NIR)")
    parser.add_argument("--control_ckpt", type=str, default="./checkpoints_rgb_only/multitask/best_model.pt",
                        help="对照组checkpoint (RGB-only)")
    parser.add_argument("--generator_ckpt", type=str, default="./checkpoints/generator/best_model.pt",
                        help="阶段一NIR生成器checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="./output/ablation")
    return parser.parse_args()


def run_eval(name, eval_args, output_dir):
    """运行一组评估(实时转发输出), 返回完整stdout文本"""
    cmd = [sys.executable, EVAL_SCRIPT] + eval_args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'=' * 60}")
    print(f"消融实验: {name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    chunks = []
    for line in proc.stdout:
        print(line, end="")
        chunks.append(line)
    proc.wait()
    text = "".join(chunks)

    # 原始输出始终存档
    with open(os.path.join(output_dir, f"raw_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    if proc.returncode != 0:
        print(f"\n[错误] 评估进程退出码 {proc.returncode}, 已中止消融实验")
        sys.exit(1)
    return text


def archive_eval_artifacts(group_dir):
    """把 evaluate_multitask.py 的共享输出归档到分组目录, 防止两组互相覆盖"""
    os.makedirs(group_dir, exist_ok=True)
    for fname in EVAL_ARTIFACTS:
        src = os.path.join(DEFAULT_EVAL_OUTPUT, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(group_dir, fname))


def parse_metrics(text):
    """从 evaluate_multitask.py 的输出中解析关键指标"""
    m = {}

    mm = re.search(r"Accuracy:\s*([\d.]+)", text)
    if not mm:
        raise ValueError("无法解析 Accuracy")
    m["accuracy"] = float(mm.group(1))

    mm = re.search(r"Macro-F1[^:\n]*:\s*([\d.]+)", text)
    if not mm:
        raise ValueError("无法解析 Macro-F1")
    m["macro_f1"] = float(mm.group(1))

    mm = re.search(r"卡路里:\s*MAE=([\d.]+)\s*kcal,\s*RMSE=([\d.]+)\s*kcal,\s*MAPE=([\d.]+)%", text)
    if not mm:
        raise ValueError("无法解析卡路里回归指标")
    m["kcal_mae"], m["kcal_rmse"], m["kcal_mape"] = (float(x) for x in mm.groups())

    mm = re.search(r"质量:\s*MAE=([\d.]+)\s*g,\s*RMSE=([\d.]+)\s*g", text)
    if not mm:
        raise ValueError("无法解析质量回归指标")
    m["mass_mae"], m["mass_rmse"] = (float(x) for x in mm.groups())

    mm = re.search(r"样本数:\s*(\d+)", text)
    m["n_samples"] = int(mm.group(1)) if mm else None

    mm = re.search(r"\(epoch=(\d+)\)", text)
    m["epoch"] = int(mm.group(1)) if mm else None

    return m


def build_report(results, split):
    """生成消融对比报告文本"""
    M = results["main"]["metrics"]
    C = results["control"]["metrics"]

    def lower_is_better(mv, cv):
        return (mv - cv) / cv * 100.0  # 负值 = 主实验更低 = 更优

    rows = [
        ("Accuracy", f"{M['accuracy']:.4f}", f"{C['accuracy']:.4f}",
         f"{(M['accuracy'] - C['accuracy']) * 100:+.2f}pp"),
        ("Macro-F1(13类)", f"{M['macro_f1']:.4f}", f"{C['macro_f1']:.4f}",
         f"{M['macro_f1'] - C['macro_f1']:+.4f}"),
        ("kcal MAE", f"{M['kcal_mae']:.1f}", f"{C['kcal_mae']:.1f}",
         f"{lower_is_better(M['kcal_mae'], C['kcal_mae']):+.1f}%"),
        ("kcal RMSE", f"{M['kcal_rmse']:.1f}", f"{C['kcal_rmse']:.1f}",
         f"{lower_is_better(M['kcal_rmse'], C['kcal_rmse']):+.1f}%"),
        ("kcal MAPE", f"{M['kcal_mape']:.1f}%", f"{C['kcal_mape']:.1f}%",
         f"{M['kcal_mape'] - C['kcal_mape']:+.1f}pp"),
        ("mass MAE", f"{M['mass_mae']:.1f}", f"{C['mass_mae']:.1f}",
         f"{lower_is_better(M['mass_mae'], C['mass_mae']):+.1f}%"),
        ("mass RMSE", f"{M['mass_rmse']:.1f}", f"{C['mass_rmse']:.1f}",
         f"{lower_is_better(M['mass_rmse'], C['mass_rmse']):+.1f}%"),
    ]

    n = M.get("n_samples") or C.get("n_samples") or "?"
    lines = []
    lines.append("=" * 76)
    lines.append(f"消融实验对比报告: RGB+预测NIR vs RGB-only  ({split}集, n={n})")
    lines.append(f"主实验: {results['main']['checkpoint']} (epoch={M.get('epoch', '?')})")
    lines.append(f"对照组: {results['control']['checkpoint']} (epoch={C.get('epoch', '?')})")
    lines.append("=" * 76)
    lines.append(f"{'Metric':<16}{'RGB+PredNIR':>14}{'RGB-only':>12}{'Delta':>12}")
    lines.append("-" * 76)
    for label, mv, cv, delta in rows:
        lines.append(f"{label:<16}{mv:>14}{cv:>12}{delta:>12}")
    lines.append("-" * 76)
    lines.append("注: 回归指标(MAE/RMSE/MAPE)的负Delta=误差更小=更优; 分类指标正Delta=更优")

    kcal_impr = -lower_is_better(M['kcal_mae'], C['kcal_mae'])
    mass_impr = -lower_is_better(M['mass_mae'], C['mass_mae'])
    acc_delta = (M['accuracy'] - C['accuracy']) * 100
    lines.append("")
    lines.append(
        f"结论: 预测NIR通道使卡路里MAE降低 {kcal_impr:.1f}% "
        f"({C['kcal_mae']:.1f} -> {M['kcal_mae']:.1f} kcal), "
        f"质量MAE降低 {mass_impr:.1f}% ({C['mass_mae']:.1f} -> {M['mass_mae']:.1f} g), "
        f"Accuracy变化 {acc_delta:+.2f}pp。"
    )
    return "\n".join(lines)


def main():
    args = parse_args()

    output_dir = os.path.normpath(os.path.join(PROJECT_ROOT, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    experiments = [
        {
            "key": "main",
            "name": "1_rgb_plus_pred_nir",
            "desc": "RGB+预测NIR (主实验)",
            "checkpoint": args.main_ckpt,
            "eval_args": [
                "--config", args.config,
                "--checkpoint", args.main_ckpt,
                "--generator_ckpt", args.generator_ckpt,
                "--split", args.split,
            ],
        },
        {
            "key": "control",
            "name": "2_rgb_only",
            "desc": "RGB-only (对照组, NIR置零)",
            "checkpoint": args.control_ckpt,
            "eval_args": [
                "--config", args.config,
                "--checkpoint", args.control_ckpt,
                "--split", args.split,
            ],
        },
    ]

    results = {}
    for exp in experiments:
        text = run_eval(exp["name"], exp["eval_args"], output_dir)
        metrics = parse_metrics(text)
        archive_eval_artifacts(os.path.join(output_dir, exp["name"]))
        results[exp["key"]] = {
            "desc": exp["desc"],
            "checkpoint": exp["checkpoint"],
            "metrics": metrics,
        }

    report = build_report(results, args.split)
    print("\n" + report)

    report_path = os.path.join(output_dir, "ablation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "split": args.split,
                "experiments": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n对比报告已保存: {report_path}")
    print(f"结果JSON已保存: {json_path}")


if __name__ == "__main__":
    main()
