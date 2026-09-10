"""汇总 ms2 六个运行（RGB vs RGB+NIR × seed42/43/44）的测试指标，供写报告与决策。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print(f"{'tag':22s}{'best_ep':>8s}{'kcal MAE':>10s}{'kcal MAPE':>11s}{'mass MAE':>10s}{'neg kcal/mass':>15s}")
rows = {}
for s in (42, 43, 44):
    for m in ('rgb', 'rgbnir'):
        p = ROOT / f'results/meal_exp_ms2_{m}_seed{s}/test_metrics.json'
        if not p.exists():
            print(f'  缺失 {p}')
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        mt = d['metrics']['metrics']
        c, ms = mt['calories'], mt['mass']
        key = f'{m}_seed{s}'
        rows[key] = {'kcal_mae': c['mae'], 'kcal_mape': c.get('mape_nonzero_percent'),
                     'mass_mae': ms['mae'], 'best_epoch': d['best_epoch'],
                     'neg_kcal': c['negative_predictions'], 'neg_mass': ms['negative_predictions']}
        print(f"{'ms2_'+key:22s}{d['best_epoch']:>8d}{c['mae']:>10.2f}"
              f"{(c.get('mape_nonzero_percent') or 0):>10.2f}%{ms['mae']:>10.2f}"
              f"{str(c['negative_predictions'])+'/'+str(ms['negative_predictions']):>15s}")

print()
print('配对差 (NIR - RGB, 由 scripts/analyze_paired_nir.py 计算):')
pa = json.loads((ROOT / 'artifacts/paired-nir-ms2.json').read_text(encoding='utf-8'))
for s in ('42', '43', '44'):
    c = pa['seeds'][s]['calories']['paired']
    m = pa['seeds'][s]['mass']['paired']
    print(f"  seed{s}: kcal Δ={c['mean_delta_nir_minus_rgb']:+.2f} "
          f"CI[{c['bootstrap_per_dish']['ci_low']:+.2f},{c['bootstrap_per_dish']['ci_high']:+.2f}] "
          f"跨0={c['bootstrap_per_dish']['crosses_zero']} | "
          f"mass Δ={m['mean_delta_nir_minus_rgb']:+.2f} "
          f"CI[{m['bootstrap_per_dish']['ci_low']:+.2f},{m['bootstrap_per_dish']['ci_high']:+.2f}] "
          f"跨0={m['bootstrap_per_dish']['crosses_zero']}")
for f in ('calories', 'mass'):
    a = pa['across_seeds'][f]
    print(f"  跨 seed {f}: meanΔ={a['mean_delta']:+.2f} 范围[{a['min']:+.2f},{a['max']:+.2f}] "
          f"全部NIR更好={a['all_seeds_nir_better']} 全部RGB更好={a['all_seeds_rgb_better']}")

rgb_maes = [rows[f'rgb_seed{s}']['kcal_mae'] for s in (42, 43, 44)]
nir_maes = [rows[f'rgbnir_seed{s}']['kcal_mae'] for s in (42, 43, 44)]
print()
print(f"RGB 臂三个种子的热量 MAE: {[round(x,2) for x in rgb_maes]} → 极差 {max(rgb_maes)-min(rgb_maes):.2f} kcal")
print(f"NIR 臂三个种子的热量 MAE: {[round(x,2) for x in nir_maes]} → 极差 {max(nir_maes)-min(nir_maes):.2f} kcal")
