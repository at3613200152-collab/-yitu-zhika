"""Strict, offline inference for audited current experiments, not legacy weights."""
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from scripts.capsicum_job import file_digest
from src.training.train_meal_official import MealNet, MEAN, STD, MANIFEST, MANIFEST_SHA
from src.training.train_meal_nir_official import NirMealNet

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_ZH = {'dairy': '乳制品', 'dessert': '甜点', 'egg': '蛋类', 'grain': '谷物主食',
    'meat': '肉类', 'mixed': '混合餐食', 'other': '其他', 'sauce_condiment': '酱料调味品',
    'seafood': '水产', 'soup_stew': '汤炖菜', 'vegetable': '蔬菜'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audited_checkpoint(name, audit_name):
    folder = ROOT/'results'/name
    result, audit, protocol = [read(folder/f'{part}.json') for part in ('test_metrics', audit_name, 'protocol')]
    path = ROOT/'checkpoints'/name/'best.pt'
    digest = file_digest(path)
    if not audit['passed'] or digest != result['checkpoint_sha256'] or digest != audit['checkpoint_sha256']:
        raise ValueError('Model checkpoint failed artifact audit: '+name)
    if result['manifest_sha256'] != MANIFEST_SHA or audit['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Model uses a different meal manifest')
    if file_digest(folder/'test_predictions.csv') != audit['predictions_sha256']:
        raise ValueError('Audited predictions changed: '+name)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['config'] != protocol or checkpoint['epoch'] != result['best_epoch']:
        raise ValueError('Model selection/configuration mismatch: '+name)
    return checkpoint


class ExperimentPipeline:
    def __init__(self, device='cpu'):
        # CPU default leaves the GPU available for background training.
        self.device = torch.device(device)
        torch.set_num_threads(2)
        self.lock = threading.Lock()
        if file_digest(MANIFEST) != MANIFEST_SHA:
            raise ValueError('Frozen meal manifest changed')
        self.manifest = read(MANIFEST)
        self.categories = {v: k for k, v in self.manifest['category_to_idx'].items()}
        # 模型来源绑定：主结果（分类 + 卡路里/重量）由 NIR 模型产出，RGB 仅作对照。
        # 详见 P0-A：主结果版本模型必须与产出模型一致，避免溯源错标。
        self.target_names = ['calories', 'mass']
        self.macros_status = 'unsupported'   # 两目标模型不输出蛋白/碳水/脂肪
        self.macros_unit = 'g'
        self.rgb_version = 'meal_rgb_official_v1'
        self.nir_version = 'meal_nir_official_v1'
        import hashlib as _hl
        self.rgb_sha = _hl.sha256((ROOT / 'checkpoints/meal_rgb_official_v1/best.pt').read_bytes()).hexdigest()
        self.nir_sha = _hl.sha256((ROOT / 'checkpoints/meal_nir_official_v1/best.pt').read_bytes()).hexdigest()
        rgb = audited_checkpoint('meal_rgb_official_v1', 'paired_completion_audit')
        self.rgb = MealNet(self.manifest, pretrained=False)
        self.rgb.load_state_dict(rgb['model'], strict=True)
        del rgb
        nir = audited_checkpoint('meal_nir_official_v1', 'paired_completion_audit')
        generator = {k.removeprefix('generator.'): v for k, v in nir['model'].items() if k.startswith('generator.')}
        self.nir = NirMealNet(self.manifest, generator, pretrained=False)
        self.nir.load_state_dict(nir['model'], strict=True)
        self.rgb.to(self.device).eval()
        self.nir.to(self.device).eval()
        self.external = None
        self.external_transform = None

    def load_external_if_ready(self):
        folder = ROOT/'results/calorieclip_official_v1'
        if self.external is not None:
            return '已完成并通过审计'
        if not (folder/'completion_audit.json').exists():
            status = read(folder/'status.json') if (folder/'status.json').exists() else {}
            stage = status.get('stage', 'not_started')
            if stage in ('failed', 'paused', 'paused_budget'):
                return '任务暂停或失败，暂不提供预测'
            if stage == 'complete':
                return '训练完成，等待独立审计'
            return f"训练中，第 {status.get('epoch', 0)}/30 轮；暂不提供预测" if status else '尚未训练'
        from src.training.train_calorieclip_official import CalorieCLIP, preprocess
        checkpoint = audited_checkpoint('calorieclip_official_v1', 'completion_audit')
        self.external = CalorieCLIP()
        self.external.load_state_dict(checkpoint['model'], strict=True)
        self.external.to(self.device).eval()
        self.external_transform = preprocess()
        return '已完成并通过审计'

    def predict(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError('Expected a PIL image')
        image = image.convert('RGB')
        if min(image.size) < 1:
            raise ValueError('Empty image')
        with self.lock, torch.inference_mode():
            external_status = self.load_external_if_ready()
            tensor = TF.normalize(TF.to_tensor(image.resize((256, 256), Image.Resampling.BILINEAR)), MEAN, STD)
            tensor = tensor.unsqueeze(0).to(self.device)
            _, rgb_values = self.rgb(tensor)
            logits, nir_values = self.nir(tensor)
            rgb01 = tensor*self.nir.rgb_std+self.nir.rgb_mean
            nir_image = ((self.nir.generator(rgb01*2-1).float()+1)/2)[0, 0]
            if not all(torch.isfinite(v).all() for v in (rgb_values, logits, nir_values, nir_image)):
                raise FloatingPointError('Non-finite model output')
            probabilities = logits.float().softmax(1)[0]
            index = int(probabilities.argmax())
            # 多类别置信度分布（按置信度降序，截断到 5），供前端"食物种类"提示
            sorted_idx = torch.argsort(probabilities, descending=True)[:5].tolist()
            category_probs = [
                {
                    'idx': i,
                    'id': self.categories[i],
                    'name': CATEGORY_ZH[self.categories[i]],
                    'prob': round(float(probabilities[i]), 3),
                    'pct': round(float(probabilities[i]) * 100, 1),
                }
                for i in sorted_idx
            ]
            result = {'calories': float(nir_values[0, 0]), 'weight': float(nir_values[0, 1]),
                'rgb_calories': float(rgb_values[0, 0]), 'rgb_weight': float(rgb_values[0, 1]),
                'nir_image': np.rint(nir_image.cpu().numpy().clip(0, 1)*255).astype(np.uint8),
                'category_idx': index, 'category_name': CATEGORY_ZH[self.categories[index]],
                'category_prob': float(probabilities[index]),
                'category_probs': category_probs,
                'classification_valid': True,
                # P0-A：主结果来源模型绑定（主结果来自 NIR；RGB 仅为对照）
                'source_model': self.nir_version,
                'source_model_sha256': self.nir_sha,
                'rgb_model': self.rgb_version,
                'rgb_model_sha256': self.rgb_sha,
                'target_names': self.target_names,
                # P0-A：三大营养素——当前两目标模型不输出，值为 None + 状态，绝不伪造
                'protein_g': None, 'carbohydrate_g': None, 'fat_g': None,
                'macros_status': self.macros_status, 'macros_unit': self.macros_unit,
                'category_prob_note': '模型置信度，非识别准确率',
                'external_calories': None, 'external_status': external_status,
                'inference_precision': 'FP32', 'device': str(self.device)}
            if self.external is not None:
                value = self.external(self.external_transform(image).unsqueeze(0).to(self.device))
                if not torch.isfinite(value).all():
                    raise FloatingPointError('Non-finite external baseline output')
                result['external_calories'] = float(value.item())
            return result


def format_prediction(result):
    category = f"粗类别：{result['category_name']}（模型分数 {result['category_prob']:.1%}，未经置信度校准）"
    external = result.get('external_calories')
    external_text = f'{external:.1f} kcal' if external is not None else result['external_status']
    text = ('## 当前照片的模型估计\n\n'
        '| 模型 | 热量 | 重量 |\n|---|---:|---:|\n'
        f"| RGB 内部对照 | {result['rgb_calories']:.1f} kcal | {result['rgb_weight']:.1f} g |\n"
        f"| RGB＋预测 NIR | {result['calories']:.1f} kcal | {result['weight']:.1f} g |\n"
        f'| CalorieCLIP 类外部基线 | {external_text} | 不支持 |\n\n'
        '11 个类别由配料标签归纳，不是官方精细菜名。预测 NIR 是相对强度图，不是实测光谱。'
        '单张照片无法确定实际份量；本页面用于课程实验，不用于饮食或医疗决策。\n\n'
        '同一 507 餐盘测试集：RGB 热量 MAE 56.55 kcal；RGB＋NIR 56.21 kcal。'
        '单种子差异很小，尚无稳定提升证据。当前推理为 FP32，正式测试使用 BF16，数值可能略有差异。')
    numbers = [result[k] for k in ('calories', 'weight', 'rgb_calories', 'rgb_weight')]
    if external is not None:
        numbers.append(external)
    if any(value < 0 for value in numbers):
        text += '\n\n警告：出现不合理负值，表示模型在此图上失效；保留原始输出，不应按此结果使用。'
    return result['nir_image'], category, text
