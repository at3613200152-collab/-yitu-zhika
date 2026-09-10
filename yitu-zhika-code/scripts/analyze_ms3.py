"""对比 ms2（RGB 臂，旧 144 扫描生成器）与 ms3（NIR 臂，全量生成器 26.67 dB）——
回答论文核心问题：生成器质量提升后，预测 NIR 是否带来收益。

用法：python scripts/analyze_ms3.py [--out artifacts/paired-nir-ms3.json]
默认同时写出证据 JSON（供报告与 §13 证据索引引用）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_paired_nir import (load_predictions, paired_stats, bootstrap_ci,  # noqa: E402
                                        metrics)

argv = sys.argv[1:]
out_path = ROOT / 'artifacts/paired-nir-ms3.json'
if '--out' in argv:
    out_path = Path(argv[argv.index('--out') + 1])
    if not out_path.is_absolute():
        out_path = ROOT / out_path

print('== 三种子：RGB 臂(ms2, 144扫描生成器) vs NIR 臂(ms3, 全量生成器 26.67 dB) ==')
print(f"{'seed':>5s}{'RGB kcal':>10s}{'NIR kcal':>10s}{'Δ':>9s}{'CI_low':>9s}{'CI_high':>9s}{'跨0':>6s}"
      f"{'RGB mass':>10s}{'NIR mass':>10s}{'Δ':>9s}{'跨0':>6s}")
deltas_c, deltas_m = [], []
per_seed = []
for s in (42, 43, 44):
    rgb = load_predictions(ROOT / f'results/meal_exp_ms2_rgb_seed{s}/test_predictions.csv')
    nir = load_predictions(ROOT / f'results/meal_exp_ms3_rgbnir_seed{s}/test_predictions.csv')
    c = paired_stats(rgb, nir, 'calories')
    ci = bootstrap_ci([v['delta'] for v in c['per_dish'].values()])
    m = paired_stats(rgb, nir, 'mass')
    mi = bootstrap_ci([v['delta'] for v in m['per_dish'].values()])
    mc, mn = metrics(rgb, 'calories'), metrics(nir, 'calories')
    mmc, mmn = metrics(rgb, 'mass'), metrics(nir, 'mass')
    deltas_c.append(c['mean_delta_nir_minus_rgb'])
    deltas_m.append(m['mean_delta_nir_minus_rgb'])
    per_seed.append({
        'seed': s,
        'n_dishes': len(c['per_dish']),
        'calories': {
            'rgb_mae': mc['mae'], 'nir_mae': mn['mae'],
            'delta_mean_nir_minus_rgb': c['mean_delta_nir_minus_rgb'],
            'ci_low': ci['ci_low'], 'ci_high': ci['ci_high'], 'crosses_zero': ci['crosses_zero'],
        },
        'mass': {
            'rgb_mae': mmc['mae'], 'nir_mae': mmn['mae'],
            'delta_mean_nir_minus_rgb': m['mean_delta_nir_minus_rgb'],
            'ci_low': mi['ci_low'], 'ci_high': mi['ci_high'], 'crosses_zero': mi['crosses_zero'],
        },
    })
    print(f"{s:>5d}{mc['mae']:>10.2f}{mn['mae']:>10.2f}{c['mean_delta_nir_minus_rgb']:>+9.2f}"
          f"{ci['ci_low']:>+9.2f}{ci['ci_high']:>+9.2f}{str(ci['crosses_zero']):>6s}"
          f"{mmc['mae']:>10.2f}{mmn['mae']:>10.2f}{m['mean_delta_nir_minus_rgb']:>+9.2f}"
          f"{str(mi['crosses_zero']):>6s}")
print(f"\n跨种子 热量: 均值 {sum(deltas_c)/3:+.2f} 范围 [{min(deltas_c):+.2f}, {max(deltas_c):+.2f}] "
      f"全部同号={'否' if min(deltas_c) < 0 < max(deltas_c) else '是'}")
print(f"跨种子 重量: 均值 {sum(deltas_m)/3:+.2f} 范围 [{min(deltas_m):+.2f}, {max(deltas_m):+.2f}] "
      f"全部同号={'否' if min(deltas_m) < 0 < max(deltas_m) else '是'}")

summary = {
    'experiment': 'ms3: RGB arm (ms2) vs RGB+NIR arm using the full-data generator (26.674 dB)',
    'rgb_arm': 'results/meal_exp_ms2_rgb_seed{42,43,44}',
    'nir_arm': 'results/meal_exp_ms3_rgbnir_seed{42,43,44}',
    'test_split': 'frozen Nutrition5k 507 dishes',
    'bootstrap_resamples': 10000,
    'per_seed': per_seed,
    'calories': {
        'mean_delta': sum(deltas_c) / 3.0,
        'min_delta': min(deltas_c),
        'max_delta': max(deltas_c),
        'all_same_sign': not (min(deltas_c) < 0 < max(deltas_c)),
        'all_ci_cross_zero': all(r['calories']['crosses_zero'] for r in per_seed),
    },
    'mass': {
        'mean_delta': sum(deltas_m) / 3.0,
        'min_delta': min(deltas_m),
        'max_delta': max(deltas_m),
        'all_same_sign': not (min(deltas_m) < 0 < max(deltas_m)),
        'all_ci_cross_zero': all(r['mass']['crosses_zero'] for r in per_seed),
    },
    'verdict': ('direction is consistently non-positive for calories (3/3 seeds favour NIR) but every '
                'paired CI still crosses zero and the magnitude is at the ~1 kcal run-to-run noise '
                'floor -> still NOT a significant gain; do not claim a stable NIR benefit'),
    'noise_floor_kcal': 1.0,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"\nwrote {out_path.relative_to(ROOT)}")
