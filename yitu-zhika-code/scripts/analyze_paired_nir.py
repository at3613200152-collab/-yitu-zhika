"""RGB vs RGB+NIR 的同餐盘配对与多种子显著性分析（用于决定 NIR 是否值得部署）。

输入：`results/meal_exp/<tag>/test_predictions.csv`
（列：dish_id, true_calories, true_mass, pred_calories, pred_mass, true_coarse_class, pred_coarse_class）
可选：`--manifest` 提供 capture_day_utc，用于按拍摄日分组 bootstrap（与既有分析口径一致）。

输出（JSON 证据 + 控制台表）：
- 每个 seed：两模式各自的 MAE/RMSE/R²/非零 MAPE；逐盘 ΔMAE 的均值/中位数/改善比例；
  按餐盘重采样与按拍摄日重采样的 95% bootstrap CI；
- 跨 seed：每 seed 的 ΔMAE 及其极差；合并配对 CI；明确给出"是否跨 0"的判定。

用法：
  python scripts/analyze_paired_nir.py --seeds 42,43,44 \
      --rgb-pattern 'results/meal_exp/ms_rgb_seed{seed}/test_predictions.csv' \
      --nir-pattern 'results/meal_exp/ms_rgbnir_seed{seed}/test_predictions.csv' \
      --manifest results/meal_manifest.json
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json                      # noqa: E402

TARGETS = [('calories', 'true_calories', 'pred_calories'), ('mass', 'true_mass', 'pred_mass')]


def load_predictions(path):
    rows = {}
    with Path(path).open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows[r['dish_id']] = {
                'true_calories': float(r['true_calories']), 'true_mass': float(r['true_mass']),
                'pred_calories': float(r['pred_calories']), 'pred_mass': float(r['pred_mass']),
                'true_coarse_class': int(r['true_coarse_class']),
                'pred_coarse_class': int(r['pred_coarse_class']),
            }
    return rows


def metrics(preds, field):
    _, tk, pk = [t for t in TARGETS if t[0] == field][0]
    errs = [abs(p[pk] - p[tk]) for p in preds.values()]
    n = len(errs)
    mae = sum(errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    truths = [p[tk] for p in preds.values()]
    mean_t = sum(truths) / n
    ss_tot = sum((t - mean_t) ** 2 for t in truths)
    ss_res = sum((p[pk] - p[tk]) ** 2 for p in preds.values())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    nz = [(p[pk], p[tk]) for p in preds.values() if p[tk] != 0]
    mape = sum(abs(pp - tt) / abs(tt) for pp, tt in nz) / len(nz) * 100 if nz else None
    return {'n': n, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape_nonzero_percent': mape,
            'mape_n': len(nz), 'mae_over_mean_target_percent': mae / mean_t * 100 if mean_t else None}


def paired_stats(rgb, nir, field):
    _, tk, pk = [t for t in TARGETS if t[0] == field][0]
    ids = sorted(set(rgb) & set(nir))
    diffs, errs = [], []
    for d in ids:
        e_rgb = abs(rgb[d][pk] - rgb[d][tk])
        e_nir = abs(nir[d][pk] - nir[d][tk])
        diffs.append(e_nir - e_rgb)      # 负值 = NIR 更好
        errs.append((d, e_rgb, e_nir))
    n = len(diffs)
    mean_d = sum(diffs) / n
    srt = sorted(diffs)
    median_d = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    return {'n': n, 'mean_delta_nir_minus_rgb': mean_d, 'median_delta': median_d,
            'n_dishes_nir_better': sum(1 for x in diffs if x < 0),
            'n_dishes_rgb_better': sum(1 for x in diffs if x > 0),
            'n_ties': sum(1 for x in diffs if x == 0),
            'per_dish': {d: {'rgb_abs_err': a, 'nir_abs_err': b, 'delta': b - a} for d, a, b in errs}}


def bootstrap_ci(values, groups=None, iters=10000, seed=12345, alpha=0.05):
    rng = random.Random(seed)
    if groups is None:
        n = len(values)
        means = []
        for _ in range(iters):
            means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    else:
        keys = sorted(groups)
        means = []
        for _ in range(iters):
            picked = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
            pool = [v for k in picked for v in groups[k]]
            means.append(sum(pool) / len(pool))
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return {'mean': sum(values) / len(values), 'ci_low': lo, 'ci_high': hi,
            'crosses_zero': lo <= 0 <= hi, 'iters': iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--rgb-pattern',
                    default=str(ROOT / 'results/meal_exp_ms_rgb_seed{seed}/test_predictions.csv'))
    ap.add_argument('--nir-pattern',
                    default=str(ROOT / 'results/meal_exp_ms_rgbnir_seed{seed}/test_predictions.csv'))
    ap.add_argument('--manifest', default=str(ROOT / 'results/meal_official_v1/manifest.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    capture_day = {}
    mpath = Path(args.manifest)
    if mpath.exists():
        for r in json.loads(mpath.read_text(encoding='utf-8'))['rows']:
            capture_day[r['dish_id']] = r.get('capture_day_utc') or r.get('capture_day') or 'unknown'

    report = {'seeds': {}, 'manifest': str(mpath) if mpath.exists() else None}
    for seed in [s.strip() for s in args.seeds.split(',') if s.strip()]:
        rgb_path = Path(args.rgb_pattern.format(seed=seed))
        nir_path = Path(args.nir_pattern.format(seed=seed))
        if not rgb_path.exists() or not nir_path.exists():
            report['seeds'][seed] = {'status': 'missing', 'rgb': str(rgb_path), 'nir': str(nir_path)}
            continue
        rgb, nir = load_predictions(rgb_path), load_predictions(nir_path)
        entry = {'status': 'ok', 'rgb_csv': str(rgb_path), 'nir_csv': str(nir_path)}
        for field in ('calories', 'mass'):
            st = paired_stats(rgb, nir, field)
            per_dish = st.pop('per_dish')
            st['bootstrap_per_dish'] = bootstrap_ci([v['delta'] for v in per_dish.values()])
            if capture_day:
                groups = {}
                for d, v in per_dish.items():
                    groups.setdefault(capture_day.get(d, 'unknown'), []).append(v['delta'])
                st['bootstrap_per_capture_day'] = bootstrap_ci(
                    [v['delta'] for v in per_dish.values()], groups=groups)
                st['n_capture_days'] = len(groups)
            entry[field] = {'rgb': metrics(rgb, field), 'nir': metrics(nir, field), 'paired': st}
        report['seeds'][seed] = entry

    ok = {s: v for s, v in report['seeds'].items() if v.get('status') == 'ok'}
    if ok:
        for field in ('calories', 'mass'):
            deltas = [v[field]['paired']['mean_delta_nir_minus_rgb'] for v in ok.values()]
            report.setdefault('across_seeds', {})[field] = {
                'per_seed_delta': {s: v[field]['paired']['mean_delta_nir_minus_rgb'] for s, v in ok.items()},
                'mean_delta': sum(deltas) / len(deltas),
                'min': min(deltas), 'max': max(deltas),
                'all_seeds_nir_better': all(d < 0 for d in deltas),
                'all_seeds_rgb_better': all(d > 0 for d in deltas),
                'n_seeds': len(deltas),
            }

    out = Path(args.out) if args.out else ROOT / 'artifacts/paired-nir-analysis.json'
    atomic_json(out, report)
    print(f'证据: {out}')

    for seed, v in report['seeds'].items():
        if v.get('status') != 'ok':
            print(f'seed {seed}: 缺失 -> {v}')
            continue
        c = v['calories']
        print(f"seed {seed}:  kcal MAE rgb={c['rgb']['mae']:.3f} nir={c['nir']['mae']:.3f} "
              f"Δ={c['paired']['mean_delta_nir_minus_rgb']:+.3f} | "
              f"MAPE rgb={c['rgb']['mape_nonzero_percent']:.2f}% nir={c['nir']['mape_nonzero_percent']:.2f}% | "
              f"配对CI[{c['paired']['bootstrap_per_dish']['ci_low']:+.3f},"
              f"{c['paired']['bootstrap_per_dish']['ci_high']:+.3f}] "
              f"跨0={c['paired']['bootstrap_per_dish']['crosses_zero']} | "
              f"改善餐盘 {c['paired']['n_dishes_nir_better']}/{c['paired']['n']}")
        m = v['mass']
        print(f"          mass MAE rgb={m['rgb']['mae']:.3f} nir={m['nir']['mae']:.3f} "
              f"Δ={m['paired']['mean_delta_nir_minus_rgb']:+.3f} "
              f"CI[{m['paired']['bootstrap_per_dish']['ci_low']:+.3f},"
              f"{m['paired']['bootstrap_per_dish']['ci_high']:+.3f}] "
              f"跨0={m['paired']['bootstrap_per_dish']['crosses_zero']}")
    if 'across_seeds' in report:
        for field, a in report['across_seeds'].items():
            print(f"跨 seed {field}: meanΔ={a['mean_delta']:+.3f} 范围[{a['min']:+.3f},{a['max']:+.3f}] "
                  f"全部NIR更好={a['all_seeds_nir_better']} 全部RGB更好={a['all_seeds_rgb_better']}")


if __name__ == '__main__':
    main()
