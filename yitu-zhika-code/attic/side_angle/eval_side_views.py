"""侧视帧留出评估：在 128 道侧视菜的指定帧上比较不同五目标权重。

背景（见 docs/侧视图位置与检验说明_2026-09-10.md）：
- 这 128 道菜的 `frame_0000.png` **就是** ts 扩展清单里训练时读入的图（split=train）；
- `frame_0030/0060.png` 未参与训练，但仍是同菜同场次 → 属"帧级/视角鲁棒性"，不是新菜泛化；
- 3262 冻结清单（= 扩展清单剔除这 128 道）从未见过这些菜 → 可作**留出对照**（同一划分、同一预处理）。

用法：
  python scripts/eval_side_views.py \
      --ckpt v1_expanded:results/meal_macros_expanded_v1/manifest.json \
      --ckpt v1:results/meal_macros_v1/manifest.json \
      --frames 0000,0030,0060 --device cpu
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest          # noqa: E402
from src.models.meal_macros_net import MealMacrosNet, TARGETS      # noqa: E402
from src.training.train_meal_official import MEAN, STD             # noqa: E402

INDEX = ROOT / 'results/side_angle_index.csv'
EXPANDED = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
CKPT_ROOT = ROOT / 'checkpoints/meal_macros_v1'


def norm_frame(label):
    """帧号归一化为 4 位（兼容 PowerShell 吃掉前导零：'0' -> '0000'）。"""
    return f'{int(str(label).strip()):04d}'


def load_rows():
    rows = list(csv.DictReader(INDEX.open(encoding='utf-8')))
    exp = {r['dish_id']: r for r in json.loads(EXPANDED.read_text(encoding='utf-8'))['rows']}
    out = []
    for r in rows:
        e = exp[r['dish_id']]
        frames = {}
        for frame in r['expansion_frames'].split(';') + r['pilot_frames'].split(';'):
            if frame:
                frames[norm_frame(Path(frame).stem.split('_')[-1])] = frame
        out.append({'dish_id': r['dish_id'], 'category': r['category'],
                    'targets': e['targets'], 'mask': e['mask'],
                    'category_idx': e['category_idx'], 'frames': frames,
                    'train_image': r['train_image']})
    return out


def fmt(value, spec='.2f'):
    return 'None' if value is None else format(value, spec)


def predict(model, device, path):
    with Image.open(path) as source:
        image = source.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(device)
    with torch.inference_mode():
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, values = model(tensor)
        else:
            logits, values = model(tensor)
    values = values.detach().float().cpu()[0]
    if not torch.isfinite(values).all():
        raise FloatingPointError('Non-finite output')
    return values, int(logits.float().argmax(1).item())


def metrics(records, name):
    """records: list of dict(dish_id, target, mask, pred, pred_cls, true_cls)"""
    out = {'name': name, 'n': len(records)}
    for i, t in enumerate(TARGETS):
        errs = [abs(r['pred'][i] - r['target'][i]) for r in records if r['mask'][i] == 1]
        negs = sum(1 for r in records if r['mask'][i] == 1 and r['pred'][i] < 0)
        if not errs:
            out[t] = {'n': 0, 'mae': None, 'neg': 0}
            continue
        mae = sum(errs) / len(errs)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
        out[t] = {'n': len(errs), 'mae': mae, 'rmse': rmse, 'neg': negs}
    acc = sum(1 for r in records if r['pred_cls'] == r['true_cls']) / len(records) if records else None
    out['category_accuracy'] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True, help='tag:manifest_path')
    ap.add_argument('--frames', default='0000,0030,0060')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    frames = [norm_frame(f) for f in args.frames.split(',') if f.strip()]
    rows = load_rows()
    device = torch.device(args.device)
    report = {'frames': frames, 'n_dishes': len(rows), 'device': str(device),
              'index_csv': str(INDEX.relative_to(ROOT)), 'models': {}}

    for spec in args.ckpt:
        tag, _, manifest_path = spec.partition(':')
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        idx_to_name = {v: k for k, v in manifest['category_to_idx'].items()}
        ckpt = CKPT_ROOT / tag / 'best.pt'
        if not ckpt.exists():
            report['models'][tag] = {'status': 'missing', 'ckpt': str(ckpt)}
            print(f'[跳过] {ckpt} 不存在')
            continue
        cfg = torch.load(ckpt, map_location='cpu', weights_only=True)['config']
        if int(cfg.get('num_classes') or 0) and int(cfg['num_classes']) != len(manifest['category_to_idx']):
            raise SystemExit(f'{tag}: 权重类别数 {cfg["num_classes"]} 与 manifest {len(manifest["category_to_idx"])} 不一致')
        model = MealMacrosNet(manifest, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True)['model'], strict=True)
        model.to(device).eval()

        per_frame = {f: [] for f in frames}
        for r in rows:
            for f in frames:
                path = r['frames'].get(f)
                if not path or not Path(path).exists():
                    continue
                pred, cls = predict(model, device, path)
                per_frame[f].append({'dish_id': r['dish_id'], 'target': r['targets'],
                                     'mask': r['mask'], 'pred': [float(x) for x in pred],
                                     'pred_cls': cls, 'true_cls': r['category_idx']})
        combined = [x for f in frames if f != '0000' for x in per_frame[f]]  # 0030/0060 = 未参与训练的帧
        entry = {'status': 'ok', 'ckpt': str(ckpt.relative_to(ROOT)),
                 'ckpt_sha256': file_digest(ckpt), 'manifest': manifest_path,
                 'label_schema': manifest.get('label_schema_version', 'v1_11class'),
                 'trained_on_side_frame_0000': 'side_angle' in json.dumps(
                     [x.get('image') for x in manifest['rows'][:1]]),
                 'per_frame': {f: metrics(v, f'frame_{f}') for f, v in per_frame.items()},
                 'frames_0030_0060': metrics(combined, 'frames_0030_0060')}
        # 该权重是否把这些菜放进过训练集
        ids_in_train = {x['dish_id'] for x in manifest['rows'] if x['split'] == 'train'}
        entry['n_side_dishes_in_its_train'] = len(ids_in_train & {r['dish_id'] for r in rows})
        report['models'][tag] = entry

    report['generated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    out = Path(args.out) if args.out else ROOT / f'artifacts/side-view-eval-{time.strftime("%Y%m%d-%H%M%S")}.json'
    atomic_json(out, report)
    print(f'证据: {out}')
    for tag, e in report['models'].items():
        if e.get('status') != 'ok':
            print(f'{tag}: {e}')
            continue
        print(f"{tag}: 该权重训练集含侧视菜 {e['n_side_dishes_in_its_train']}/{report['n_dishes']}")
        for f in frames:
            m = e['per_frame'][f]
            print(f"   frame_{f} n={m['n']:3d} kcal MAE={fmt(m['calories']['mae'], '7.2f')} "
                  f"mass MAE={fmt(m['mass']['mae'], '6.2f')} 蛋白={fmt(m['protein']['mae'], '5.2f')} "
                  f"碳水={fmt(m['carbohydrate']['mae'], '5.2f')} 脂肪={fmt(m['fat']['mae'], '5.2f')} "
                  f"类别acc={fmt(m['category_accuracy'], '.3f')}")
        c = e['frames_0030_0060']
        print(f"   0030+0060 合并 n={c['n']:3d} kcal MAE={fmt(c['calories']['mae'], '7.2f')} "
              f"mass MAE={fmt(c['mass']['mae'], '6.2f')} 类别acc={fmt(c['category_accuracy'], '.3f')}")


if __name__ == '__main__':
    main()
