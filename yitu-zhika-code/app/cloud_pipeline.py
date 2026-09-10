"""Primary production predictions only; research baselines stay in the desktop demo."""
import json
import threading
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from app.experiment_pipeline import (
    ExperimentPipeline, ROOT, MACROS_CKPT, CATEGORY_CKPT, CATEGORY_ZH, MEAN, STD,
)
from scripts.capsicum_job import file_digest


class CloudPipeline(ExperimentPipeline):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        torch.set_num_threads(2)
        self.lock = threading.Lock()
        expected = json.loads((ROOT / 'runtime_hashes.json').read_text())
        for path in (MACROS_CKPT, CATEGORY_CKPT):
            relative = path.relative_to(ROOT).as_posix()
            if file_digest(path) != expected[relative]:
                raise RuntimeError('Runtime weight hash mismatch: ' + relative)
        self.macros = self.category_model = None
        self.rgb_version = self.rgb_sha = self.nir_version = self.nir_sha = None
        self.macros_unit = 'g'
        self.target_names = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']
        self.load_macros_if_ready()
        self.load_category_if_ready()
        self.manifest = json.loads((ROOT / self.category_manifest_path).read_text(encoding='utf-8'))

    def predict(self, image):
        with self.lock, torch.inference_mode():
            tensor = TF.normalize(TF.to_tensor(image.convert('RGB').resize(
                (256, 256), Image.Resampling.BILINEAR)), MEAN, STD).unsqueeze(0).to(self.device)
            _, values = self.macros(tensor)
            logits, _ = self.category_model(tensor)
            if not torch.isfinite(values).all() or not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite cloud model output')
            prob = logits.float().softmax(1)[0]
            index = int(prob.argmax())
            top = torch.argsort(prob, descending=True)[:5].tolist()
            result = dict(zip(['calories', 'weight', 'protein_g', 'carbohydrate_g', 'fat_g'],
                              [float(v) for v in values[0]]))
            result.update(
                category_idx=index, category_name=CATEGORY_ZH[self.category_names[index]],
                category_prob=float(prob[index]), classification_valid=True,
                category_probs=[{'idx': i, 'id': self.category_names[i],
                    'name': CATEGORY_ZH[self.category_names[i]], 'prob': round(float(prob[i]), 3),
                    'pct': round(float(prob[i])*100, 1)} for i in top],
                source_model=self.macros_version, source_model_sha256=self.macros_sha,
                category_model=self.category_version, category_model_sha256=self.category_sha,
                category_label_schema=self.category_label_schema, label_schema=self.category_label_schema,
                category_manifest=self.category_manifest_path, macros_status='supported', macros_unit='g',
                target_names=self.target_names, inference_precision='FP32', device=str(self.device),
                rgb_model=None, rgb_model_sha256=None, external_calories=None,
                external_status='not_deployed_research_baseline',
                category_prob_note='模型置信度，非识别准确率')
            return result
