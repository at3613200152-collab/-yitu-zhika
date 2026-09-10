"""端到端一致性自检（整改清单 §五 验收）：
同一固定图片，离线 ExperimentPipeline 与 HTTP /predict 的数值/版本/单位必须一致。

- 离线：ExperimentPipeline(device='cpu').predict(image)
- HTTP：POST {base}/predict (X-API-Key)
- 比对：calories/weight(1位小数)、protein_g/carbohydrate_g/fat_g、category_name、
        model_version、macros_status、target_names
写入 artifacts/inference-consistency-<stamp>/evidence.json。

用法：python scripts/check_inference_consistency.py --image test_images/test_apple_pie.jpg
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOL = 1e-6


def http_predict(base, key, image_path):
    boundary = '----dsh' + uuid.uuid4().hex
    data = Path(image_path).read_bytes()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="image"; filename="{Path(image_path).name}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + data + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    req = urllib.request.Request(base.rstrip('/') + '/predict', data=body, method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('X-API-Key', key)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode('utf-8'))


def approx(a, b, tol=TOL):
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(tol, 0.05)  # HTTP 端四舍五入到 1 位
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='test_images/test_apple_pie.jpg')
    ap.add_argument('--base', default='http://127.0.0.1:8000')
    ap.add_argument('--key', default='dev-key-change-in-prod')
    args = ap.parse_args()

    from app.experiment_pipeline import ExperimentPipeline
    from PIL import Image

    image = Image.open(ROOT.parent / args.image).convert('RGB')
    pipe = ExperimentPipeline(device='cpu')
    off = pipe.predict(image)
    on = http_predict(args.base, args.key, ROOT.parent / args.image)

    checks = {}
    for k in ('category_name', 'model_version', 'macros_status'):
        checks[k] = {'offline': off.get(k if k != 'model_version' else 'source_model'), 'http': on.get(k)}
        checks[k]['pass'] = checks[k]['offline'] == checks[k]['http']
    for k in ('calories', 'weight', 'protein_g', 'carbohydrate_g', 'fat_g'):
        o, h = off.get(k), on.get(k)
        checks[k] = {'offline': o, 'http': h, 'pass': approx(o, h)}
    # 单位/契约
    checks['macros_unit'] = {'offline': off.get('macros_unit'), 'http': on.get('macros_unit'),
                             'pass': off.get('macros_unit') == on.get('macros_unit')}
    checks['target_names'] = {'offline': off.get('target_names'), 'http': on.get('target_names'),
                              'pass': off.get('target_names') == on.get('target_names')}
    # 类别来源（决策 C：类别与热量/宏量解耦，需一并核对来源标注）
    for k in ('category_model', 'category_label_schema', 'category_manifest', 'label_schema'):
        checks[k] = {'offline': off.get(k), 'http': on.get(k), 'pass': off.get(k) == on.get(k)}
    # 异常字段一致性
    checks['abnormal_fields'] = {'offline': [k for k in ('calories','weight','protein_g','carbohydrate_g','fat_g')
                                             if isinstance(off.get(k),(int,float)) and off[k] < 0],
                                 'http': on.get('abnormal_fields', []),
                                 'pass': True}
    checks['abnormal_fields']['pass'] = checks['abnormal_fields']['offline'] == checks['abnormal_fields']['http']

    ok = all(v['pass'] for v in checks.values())
    evidence = {'image': args.image, 'base': args.base, 'all_pass': ok, 'checks': checks}
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = ROOT / f'artifacts/inference-consistency-{stamp}/evidence.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'all_pass': ok, 'evidence': str(out)}, ensure_ascii=False))
    for k, v in checks.items():
        print(f"  {k:18s} offline={v['offline']} http={v['http']} pass={v['pass']}")
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
