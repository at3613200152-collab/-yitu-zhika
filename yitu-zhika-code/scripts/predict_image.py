"""对单张图片跑在线同款推理（离线），用于人工检验任意图片（含 Nutrition5k 侧视帧）。

输出与 /predict 一致的关键字段：热量/重量/三大宏量/粗类别 + 来源模型与标签体系 + 异常标记。
默认 device=cpu（与在线服务一致，数值可与接口对齐）；GPU 空闲时可用 --device cuda。

用法：
  python scripts/predict_image.py --image "D:/yitu-data/Nutrition5k/side_angle_pilot_v1/dish_1551391337/frame_0000.png"
  python scripts/predict_image.py --image test_images/test_apple_pie.jpg --json
批量（一个目录或一批路径）：
  python scripts/predict_image.py --image <p1> --image <p2> ...
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ['calories', 'weight', 'protein_g', 'carbohydrate_g', 'fat_g',
          'category_name', 'category_prob', 'source_model', 'category_model',
          'label_schema', 'macros_status', 'abnormal_fields', 'values_note']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', action='append', required=True, help='图片路径（可重复）')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--json', action='store_true', help='只输出 JSON')
    args = ap.parse_args()

    from PIL import Image
    from app.experiment_pipeline import ExperimentPipeline

    pipe = ExperimentPipeline(device=args.device)
    out = []
    for raw in args.image:
        path = Path(raw)
        if not path.exists():
            print(f'[跳过] 文件不存在: {path}', file=sys.stderr)
            continue
        result = pipe.predict(Image.open(path).convert('RGB'))
        row = {'image': str(path), 'size': Image.open(path).size}
        for k in FIELDS:
            if k in result:
                row[k] = result[k]
        row['top5_categories'] = [(c['name'], c['pct']) for c in result.get('category_probs', [])]
        out.append(row)
        if not args.json:
            print(f"== {path.name} ({path.parent.name}) ==")
            print(f"   热量 {result['calories']:.1f} kcal | 重量 {result['weight']:.1f} g | "
                  f"类别 {result['category_name']}（{result['category_prob']:.1%}，非准确率）")
            print(f"   蛋白 {result['protein_g']:.2f} g | 碳水 {result['carbohydrate_g']:.2f} g | "
                  f"脂肪 {result['fat_g']:.2f} g")
            print(f"   来源: 回归/宏量={result['source_model']} 类别={result.get('category_model')}"
                  f"（{result.get('label_schema')}）")
            if result.get('abnormal_fields'):
                print(f"   ⚠ 异常字段: {result['abnormal_fields']}（仅作记录参考，不作正常摄入）")
            print(f"   前5类: {row['top5_categories']}")
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
