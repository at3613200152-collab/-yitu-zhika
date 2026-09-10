"""生成器两版（legacy vs hsi_unet_v2）的划分重叠检查。

目的：判断"在同一批样本上比较两版 PSNR"是否可行。
- legacy 划分：`src/data/hsi_dataset.HSIFoodIngrDataset` 的规则（子目录+hdr 排序，末尾 15% 作 val）；
- 新版划分：`data/hsi_prepared_v2/manifest.json` 的 `split`（按拍摄日分组，train/val/test）。

输出：各集合大小、交集、"两版都未见过"的可用样本数。
用法：python scripts/check_generator_split_overlap.py
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json      # noqa: E402

LEGACY_ROOT = ROOT / 'data/HSIFoodIngr-64'
V2_MANIFEST = ROOT / 'data/hsi_prepared_v2/manifest.json'


def legacy_split(val_ratio=0.15):
    """完全复刻 HSIFoodIngrDataset 的枚举顺序。"""
    samples = []
    for subdir in sorted(os.listdir(LEGACY_ROOT)):
        sub_path = LEGACY_ROOT / subdir
        if not sub_path.is_dir():
            continue
        for hdr in sorted(glob.glob(str(sub_path / '*.hdr'))):
            raw = hdr.replace('.hdr', '.raw')
            if not os.path.exists(raw):
                raw = hdr.replace('.hdr', '.dat')
            if os.path.exists(raw):
                samples.append(Path(hdr).stem)
    n = len(samples)
    n_val = int(n * val_ratio)
    return {'n': n, 'val_ratio': val_ratio, 'train': set(samples[:n - n_val]),
            'val': set(samples[n - n_val:])}


def main():
    legacy = legacy_split()
    v2_rows = json.loads(V2_MANIFEST.read_text(encoding='utf-8'))['rows']
    v2 = {'train': set(), 'val': set(), 'test': set()}
    for r in v2_rows:
        v2[r['split']].add(r['id'])

    all_ids = legacy['train'] | legacy['val']
    report = {
        'legacy': {k: (sorted(v) if isinstance(v, set) else v) for k, v in legacy.items()},
        'v2': {k: sorted(v) for k, v in v2.items()},
        'counts': {'legacy_train': len(legacy['train']), 'legacy_val': len(legacy['val']),
                   'v2_train': len(v2['train']), 'v2_val': len(v2['val']), 'v2_test': len(v2['test']),
                   'total_ids_legacy': len(all_ids), 'total_ids_v2': sum(len(v) for v in v2.values())},
        'id_sets_identical': all_ids == (v2['train'] | v2['val'] | v2['test']),
        'overlaps': {
            'v2_test ∩ legacy_train (新版测试被 legacy 训练见过)': sorted(v2['test'] & legacy['train']),
            'v2_test ∩ legacy_val': sorted(v2['test'] & legacy['val']),
            'legacy_val ∩ v2_train (legacy 的验证被新版训练见过)': sorted(legacy['val'] & v2['train']),
            'legacy_val ∩ v2_val': sorted(legacy['val'] & v2['val']),
            'legacy_val ∩ v2_test': sorted(legacy['val'] & v2['test']),
        },
    }
    report['usable_for_same_protocol'] = {
        'legacy_val_not_in_v2_train': sorted(legacy['val'] - v2['train']),
        'note': '这批样本两版都未用于训练（legacy 视其为验证、且不在新版训练集内）',
    }
    report['usable_for_same_protocol']['n'] = len(report['usable_for_same_protocol']['legacy_val_not_in_v2_train'])

    atomic_json(ROOT / 'artifacts/generator-split-overlap.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('legacy', 'v2')},
                     ensure_ascii=False, indent=2))
    print('各集合样本数:', report['counts'])
    print('两版 ID 集合是否一致:', report['id_sets_identical'])
    print('可直接用于同协议比较的样本数:', report['usable_for_same_protocol']['n'])


if __name__ == '__main__':
    main()
