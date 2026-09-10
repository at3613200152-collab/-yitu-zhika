"""阶段四 6.1 核对：扩展 manifest 真实性 / 128 餐盘是否只进训练 / 是否有划分泄漏。

只做事实核对，不做任何训练或重写；输出 JSON 结论，便于写进报告。
用法：
  python scripts/audit_expanded_manifest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import file_digest

EXPANDED = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
BASE = ROOT / 'results/meal_macros_v1/manifest.json'
CKPT_DIR = ROOT / 'checkpoints/meal_macros_v1/v1_expanded'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def split_ids(manifest):
    out = {'train': set(), 'val': set(), 'test': set()}
    for r in manifest['rows']:
        out.setdefault(r['split'], set()).add(r['dish_id'])
    return out


def main():
    report = {'expanded_manifest': str(EXPANDED.relative_to(ROOT)),
              'expanded_sha256': file_digest(EXPANDED)}
    exp, base = load(EXPANDED), load(BASE)
    idc = {'train': {}, 'val': {}, 'test': {}}
    for r in exp['rows']:
        idc[r['split']][r['dish_id']] = idc[r['split']].get(r['dish_id'], 0) + 1
    report['counts_declared'] = exp.get('counts')
    report['counts_actual'] = {k: len(v) for k, v in idc.items()}
    report['counts_actual_total'] = sum(len(v) for v in idc.values())

    dup_in_split = {k: sorted(d for d, c in v.items() if c > 1) for k, v in idc.items()}
    report['duplicate_dish_id_within_split'] = {k: v for k, v in dup_in_split.items() if v}

    sp = split_ids(exp)
    overlap = {}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        o = sorted(sp[a] & sp[b])
        if o:
            overlap[f'{a}&{b}'] = o
    report['split_overlap'] = overlap

    base_ids = set(split_ids(base)['train']) | set(split_ids(base)['val']) | set(split_ids(base)['test'])
    new_ids = {r['dish_id'] for r in exp['rows']} - base_ids
    report['new_dish_count_actual'] = len(new_ids)
    report['new_dish_count_declared'] = exp.get('new_dishes_from_expanded')
    report['new_dishes_by_split'] = {
        s: len(new_ids & sp[s]) for s in ('train', 'val', 'test')}
    report['new_dishes_only_in_train'] = set(report['new_dishes_by_split']) == {'train'} or \
        (report['new_dishes_by_split']['val'] == 0 and report['new_dishes_by_split']['test'] == 0)
    report['base_ids_missing_from_expanded'] = sorted(base_ids - {r['dish_id'] for r in exp['rows']})

    # 像素/文件哈希跨 split 重复 = 近似重复泄漏信号
    hashes = {}
    for r in exp['rows']:
        for key in ('pixel_sha256', 'image_sha256'):
            h = r.get(key)
            if h:
                hashes.setdefault((key, h), set()).add(r['split'])
    cross = {f'{k[0]}:{k[1][:12]}': sorted(v) for k, v in hashes.items() if len(v) > 1}
    report['cross_split_identical_hashes'] = cross

    # 记录的训练配置里写的是哪个 manifest（证明真的读了扩展 manifest，而不是只建了同名目录）
    report['checkpoint_manifest_binding'] = {}
    for name in ('best.pt', 'last.pt'):
        p = CKPT_DIR / name
        if not p.exists():
            report['checkpoint_manifest_binding'][name] = 'missing'
            continue
        try:
            import torch
            payload = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as exc:  # noqa: BLE001
            report['checkpoint_manifest_binding'][name] = f'load_failed: {exc}'
            continue
        cfg = (payload.get('config') or {})
        rec = cfg.get('manifest_sha256')
        report['checkpoint_manifest_binding'][name] = {
            'recorded_manifest_sha256': rec,
            'matches_expanded': rec == report['expanded_sha256'],
            'matches_base': rec == file_digest(BASE),
            'epochs_planned': cfg.get('epochs'), 'batch': cfg.get('batch'),
            'seed': cfg.get('seed'), 'protocol': cfg.get('protocol'),
            'code_sha256': cfg.get('code_sha256'),
            'keys': sorted(payload.keys()),
        }

    # 加载器是否消费多视角（诚实说明）
    sample = exp['rows'][0]
    report['loader_views'] = {
        'row_has_views_key': 'views' in sample,
        'row_keys': sorted(sample.keys()),
        'note': '五目标 MacrosDataset 继承 MealDataset，只读单张俯拍 image 字段；manifest 未提供 views，故不存在多视角训练。',
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
