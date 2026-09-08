"""Exercise the running local upload endpoint with a training image, not test tuning."""
import json
from pathlib import Path
import sys

from gradio_client import Client, handle_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest


def run():
    manifest = json.loads((ROOT/'results/meal_official_v1/manifest.json').read_text(encoding='utf-8'))
    row = next(row for row in manifest['rows'] if row['split'] == 'train')
    client = Client('http://127.0.0.1:7860', verbose=False)
    output = client.predict(handle_file(str(ROOT/row['image'])), api_name='/analyze_food')
    assert len(output) == 3 and output[0] is not None
    assert '分析失败' not in output[1] and '粗类别' in output[1]
    assert 'RGB＋预测 NIR' in output[2] and 'CalorieCLIP' in output[2]
    with Image.open(output[0]) as image:
        assert image.size == (256, 256)
    audit = {'passed': True, 'endpoint': 'http://127.0.0.1:7860/analyze_food',
        'training_dish': row['dish_id'], 'input_sha256': file_digest(ROOT/row['image']),
        'nir_size': [256, 256], 'category': output[1], 'comparison_markdown': output[2]}
    atomic_json(ROOT/'results/demo_smoke_v1.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
