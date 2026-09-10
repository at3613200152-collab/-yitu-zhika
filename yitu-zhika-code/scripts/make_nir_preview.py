"""生成 NIR 预览图：RGB / 实测 NIR / 我们预测 NIR / 队友预测 NIR(对齐) / 误差热图。

用途：直观回答"我们训练出来的 NIR 长什么样"，并顺带展示两版生成器的差距与误差分布。
样本取自队友 test 划分导出的 327 张（与评估同源、同口径）。
用法：python scripts/make_nir_preview.py --n 5
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.generator import UNetGenerator as OurUNet      # noqa: E402

DATA = ROOT / 'artifacts/hsi_v2_test'
THEIR_CKPT = Path(r'C:\Users\user\Desktop\新建文件夹\model_v2\checkpoints\generator\best_model.pt')
OUR_CKPT = ROOT / 'checkpoints/hsi_unet_v2/best.pt'
THEIR_UNET = ROOT / 'artifacts/v2_code/models_generator_unet.py'
CAL = ROOT / 'artifacts/v2_nir_calibration_rot.json'
OUT_DIR = ROOT / 'results/nir_preview'
CELL = 224
HEADER = 22
LABEL = 18


def load_models():
    spec = importlib.util.spec_from_file_location('their_unet', THEIR_UNET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    theirs = mod.UNetGenerator(input_channels=3, output_channels=1, base_features=64)
    sd = torch.load(THEIR_CKPT, map_location='cpu', weights_only=False)['model_state_dict']
    theirs.load_state_dict({k[len('generator.'):]: v for k, v in sd.items() if k.startswith('generator.')},
                           strict=True)
    theirs.eval()
    ours = OurUNet(in_channels=3, out_channels=1, base_filters=64)
    ours.load_state_dict(torch.load(OUR_CKPT, map_location='cpu', weights_only=True)['G_state_dict'], strict=True)
    ours.eval()
    return theirs, ours


@torch.inference_mode()
def predict(model, rgb01):
    x = torch.from_numpy(np.ascontiguousarray(rgb01.transpose(2, 0, 1))[None]).float() * 2 - 1
    y = ((model(x) + 1) / 2).clamp(0, 1)[0, 0].numpy()
    return y


def to_img(arr01):
    return Image.fromarray(np.clip(arr01 * 255, 0, 255).astype(np.uint8)).convert('RGB')


def error_map(pred, tgt):
    e = np.abs(pred - tgt)
    e = e / max(e.max(), 1e-6)
    # 蓝→黄 上色，便于看误差分布
    r = (np.clip(e * 2, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(e * 2 - 0.5, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1 - e * 2, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([r, g, b]).astype(np.uint8))


def psnr(pred, tgt):
    mse = float(np.mean((pred - tgt) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--worst', action='store_true',
                    help='按我们生成器 PSNR 从低到高挑选样本（展示失败案例，避免只挑好看的）')
    args = ap.parse_args()

    cal = json.loads(CAL.read_text(encoding='utf-8'))
    slope, intercept = cal['slope'], cal['intercept']
    DATA.mkdir(parents=True, exist_ok=True)
    all_files = sorted(DATA.glob('*_rgb.npy'))
    if args.worst:
        cmp_path = ROOT / 'artifacts/generator-aligned-comparison.json'
        info = json.loads(cmp_path.read_text(encoding='utf-8'))['per_sample']
        ranked = sorted(info, key=lambda r: r['our_144']['psnr'])
        picked = [r['sample'] for r in ranked[:args.n]]
        files = [f for f in all_files if f.stem.replace('_rgb', '') in set(picked)]
    else:
        files = all_files[args.start:args.start + args.n]
    theirs, ours = load_models()

    cols = ['RGB', 'Measured NIR', 'Ours (23 dB)', 'Theirs (14 dB)', '|err| Ours', '|err| Theirs']
    W = CELL * len(cols)
    H = HEADER + (CELL + LABEL * 2) * len(files)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for j, c in enumerate(cols):
        d.text((j * CELL + 6, 4), c, fill=(0, 0, 0))

    for i, f in enumerate(files):
        rgb01 = np.load(f).astype(np.float32) / 255.0
        nir16 = np.load(str(f).replace('_rgb.npy', '_nir.npy')).astype(np.float32)
        tgt = np.clip(slope * np.rot90(nir16, 3) + intercept, 0, 1)      # 我们的口径
        po = predict(ours, rgb01)
        pt = np.rot90(predict(theirs, rgb01), 3)                          # 对齐到我们口径
        row = [to_img(rgb01), to_img(tgt), to_img(po), to_img(pt),
               error_map(po, tgt), error_map(pt, tgt)]
        y0 = HEADER + i * (CELL + LABEL * 2)
        for j, im in enumerate(row):
            canvas.paste(im.resize((CELL, CELL), Image.BILINEAR), (j * CELL, y0))
        ps_o, ps_t = psnr(po, tgt), psnr(pt, tgt)
        d.text((6, y0 + CELL + 2), f'{f.stem.replace("_rgb", "")}   PSNR: ours {ps_o:.2f} dB'
                                   f'  |  theirs {ps_t:.2f} dB', fill=(0, 0, 0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'worst{len(files)}' if args.worst else f'{args.start}_{args.start + len(files) - 1}'
    out = OUT_DIR / f'nir_compare_{tag}.png'
    canvas.save(out)
    print(f'已生成: {out}  ({canvas.width}x{canvas.height})')
    print('口径: 实测 NIR = 队友 uint16 经标定仿射 rot90x3 对齐后的我们口径 [0,1]')
    print('对照: 两版生成器同一批样本、同一指标实现')


if __name__ == '__main__':
    main()
