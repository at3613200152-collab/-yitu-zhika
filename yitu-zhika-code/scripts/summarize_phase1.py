"""汇总阶段一全量数据生成器的三种子结果，并与小数据版/队友版/论文参考对照。"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print('阶段一：全量数据生成器（2772 训练样本）三种子，327 张测试集')
print(f"{'tag':14s}{'best_ep':>9s}{'PSNR(per-img)':>15s}{'PSNR(batch-MSE)':>17s}{'SSIM':>9s}{'L1':>10s}")
ps, ss = [], []
for s in (42, 43, 44):
    d = json.loads((ROOT / f'results/hsi_full_v3/full_seed{s}/test_metrics.json').read_text(encoding='utf-8'))
    t = d['test']
    ps.append(t['psnr_db']); ss.append(t['ssim'])
    print(f"full_seed{s:<5d}{d['best_epoch']:>9d}{t['psnr_db']:>15.3f}"
          f"{d['test_psnr_batch_mse_db']:>17.3f}{t['ssim']:>9.4f}{t['l1_01']:>10.5f}")
print(f"{'均值':14s}{'':>9s}{st.mean(ps):>15.3f}{'':>17s}{st.mean(ss):>9.4f}")
print(f"极差: PSNR {max(ps)-min(ps):.3f} dB, SSIM {max(ss)-min(ss):.4f}")
print()
print('对照:')
print('  路径 A-小数据(144 扫描, 93 训练)  22.84 dB / SSIM 0.837')
print('  路径 B-队友(全量但朝向错位)        14.25 dB / SSIM 0.622')
print('  论文参考(作者自采数据, 口径不同)   30.61 dB / SSIM 0.865')
