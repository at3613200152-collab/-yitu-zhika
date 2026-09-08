"""Post-training artifact audit and generated three-model comparison report."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, check_stop, file_digest, job_lock
from src.training.train_calorieclip_official import CalorieCLIP, OUTPUT, WEIGHTS, MANIFEST, MANIFEST_SHA


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(wait=False):
    torch.set_num_threads(2)
    with job_lock(OUTPUT/'audit.lock'):
        deadline = time.monotonic()+3600
        while not (OUTPUT/'test_metrics.json').exists():
            check_stop(OUTPUT)
            state = read(OUTPUT/'status.json')
            if state.get('stage') in ('failed', 'paused', 'paused_budget'):
                raise RuntimeError('Training stopped: '+state['stage'])
            if not wait or time.monotonic() > deadline:
                raise TimeoutError('Test artifact not ready within audit wait budget')
            time.sleep(10)
        manifest = read(MANIFEST)
        result, protocol, history = [read(OUTPUT/f'{name}.json') for name in ('test_metrics', 'protocol', 'epochs')]
        assert file_digest(MANIFEST) == MANIFEST_SHA == result['manifest_sha256'] == protocol['manifest_sha256']
        assert result['checkpoint_sha256'] == file_digest(WEIGHTS/'best.pt')
        assert result['predictions_sha256'] == file_digest(OUTPUT/'test_predictions.csv')
        best = torch.load(WEIGHTS/'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(WEIGHTS/'last.pt', map_location='cpu', weights_only=True)
        assert last['training_complete'] and last['epoch'] == len(history) == result['completed_epochs'] == 30
        assert [row['epoch'] for row in history] == list(range(1, 31)) and last['history'] == history
        selected = min(history, key=lambda row: row['val']['mae'])['epoch']
        assert selected == best['epoch'] == last['best_epoch'] == result['best_epoch']
        assert best['config'] == last['config'] == protocol
        assert best['target_names'] == last['target_names'] == protocol['target_names'] == ['calories']
        assert protocol['target_units'] == ['kcal'] and not protocol['food_finetuned_weights_used']
        for path, digest in protocol['code_hashes'].items():
            assert file_digest(ROOT/path) == digest
        model = CalorieCLIP()
        model.load_state_dict(best['model'], strict=True)
        for checkpoint in (best, last):
            assert all(torch.isfinite(value).all().item() for value in checkpoint['model'].values())
        del model, best, last
        with (OUTPUT/'test_predictions.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        test = {r['dish_id']: r for r in manifest['rows'] if r['split'] == 'test'}
        assert len(rows) == len(test) == len({r['dish_id'] for r in rows}) == 507
        assert {r['dish_id'] for r in rows} == set(test)
        y = np.asarray([float(r['true_calories']) for r in rows])
        p = np.asarray([float(r['pred_calories']) for r in rows])
        reference = np.asarray([test[r['dish_id']]['targets'][0] for r in rows], dtype=np.float32).astype(np.float64)
        np.testing.assert_array_equal(y, reference)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        error, nonzero = p-y, y > 0
        recomputed = {'n': len(y), 'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error**2).mean())),
            'r2': float(1-(error**2).sum()/((y-y.mean())**2).sum()),
            'mape_nonzero_percent': float(np.abs(error[nonzero]/y[nonzero]).mean()*100),
            'mape_n': int(nonzero.sum()), 'negative_predictions': int((p < 0).sum()),
            'mae_over_mean_target_percent': float(np.abs(error).mean()/y.mean()*100)}
        for key, value in recomputed.items():
            np.testing.assert_allclose(value, result['metrics'][key], rtol=1e-10, atol=1e-10)
        audit = {'passed': True, 'completed_epochs': 30, 'selected_epoch': selected,
            'checkpoint_sha256': result['checkpoint_sha256'], 'manifest_sha256': MANIFEST_SHA,
            'predictions_sha256': result['predictions_sha256'], 'recomputed_metrics': recomputed,
            'strict_architecture_load': True, 'test_rows': len(rows),
            'method': 'CPU hash, checkpoint, validation-selection and CSV checks; no further test inference or tuning.'}
        atomic_json(OUTPUT/'completion_audit.json', audit)
        paired = read(ROOT/'results/paired_comparison_v1.json')
        entries = []
        for label, name in [('RGB 内部对照', 'meal_rgb_official_v1'), ('RGB＋预测 NIR', 'meal_nir_official_v1')]:
            saved = paired['models'][name]
            entries.append({'name': label, 'run': name, 'best_epoch': saved['selected_epoch'],
                'calories': saved['metrics']['metrics']['calories'], 'mass': saved['metrics']['metrics']['mass'],
                'coarse_accuracy': saved['metrics']['coarse_category_accuracy']})
        entries.append({'name': 'CalorieCLIP 类外部基线', 'run': 'calorieclip_official_v1',
            'best_epoch': selected, 'calories': recomputed, 'mass': None, 'coarse_accuracy': None})
        atomic_json(ROOT/'results/experiment_comparison_v1.json', {'manifest_sha256': MANIFEST_SHA,
            'test_dishes': 507, 'entries': entries, 'all_completion_audits_passed': True,
            'limitations': protocol['limitations']})
        lines = ['# 三模型实验对比（自动汇总）', '',
            '同一冻结 Nutrition5k 本地俯视图子集：训练 2188 / 验证 567 / 测试 507。模型仅按验证集选择。', '',
            '| 模型 | 最佳轮 | 热量 MAE (kcal) | 热量 RMSE (kcal) | 非零 MAPE (%) | 重量 MAE (g) | 粗类别准确率 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in entries:
            calorie = row['calories']
            mass = f"{row['mass']['mae']:.3f}" if row['mass'] else '不支持'
            accuracy = f"{row['coarse_accuracy']:.2%}" if row['coarse_accuracy'] is not None else '不支持'
            lines.append(f"| {row['name']} | {row['best_epoch']} | {calorie['mae']:.3f} | {calorie['rmse']:.3f} | {calorie['mape_nonzero_percent']:.3f} | {mass} | {accuracy} |")
        lines.extend(['', '## 解读边界', '',
            '- RGB/NIR 为同骨干、同初始化共享部分、同训练预算的内部消融；外部基线使用不同骨干、预处理和损失，不隔离 NIR 变量。',
            '- RGB＋NIR 热量 MAE 差为 -0.340 kcal；按 64 个拍摄日分组的 10000 次 bootstrap 区间为 [-3.091, 2.503] kcal，跨过 0。MAPE 反而变差，尚无稳定增益证据。',
            '- 所有模型只有一个随机种子。区间条件于固定权重，不涵盖重新训练的不确定性。',
            '- 外部基线独立实现作者公开结构/config；未发布的训练细节采用已记录的本地选择，不是原论文精确复现，不直接对比作者另一测试集成绩。',
            '- 新增 128 餐盘 / 384 帧保留为单独扩容集，未混入这三次训练。',
            '- 零热量样本保留在 MAE/RMSE 中；仅 MAPE 分母为零时排除。负预测不裁剪。',
            '- HSI 测试：33 扫描，PSNR 23.333 dB / SSIM 0.81198；属于相对强度预测，不是实测 NIR。', '',
            '指标与权重哈希见各 results/<run>/completion_audit.json 或 paired_completion_audit.json。'])
        # Generated report artifact, not a hand-maintained source file.
        (ROOT/'results/experiment_comparison_v1.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    run(parser.parse_args().wait)
