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
from src.models.meal_macros_net import MealMacrosNet

ROOT = Path(__file__).resolve().parents[1]
# 五头宏量模型（P1-B）：接入在线 predict 后，主输出(热量/重量/类别/宏量)来自此模型
MACROS_VERSION = 'meal_macros_v1'
MACROS_CKPT = ROOT / 'checkpoints/meal_macros_v1/v1_expanded/best.pt'   # P1-C: 更优的扩展(3390)版
MACROS_MANIFEST = ROOT / 'results/meal_macros_expanded_v1/manifest.json'
CATEGORY_ZH = {'dairy': '乳制品', 'dessert': '甜点', 'egg': '蛋类', 'fruit': '水果',
    'grain': '谷物主食', 'meat': '肉类', 'mixed': '混合餐食', 'other': '其他',
    'sauce_condiment': '酱料调味品', 'seafood': '水产', 'soup_stew': '汤炖菜',
    'vegetable': '蔬菜'}


def resolve_macros_manifest(checkpoint):
    """解析五头权重对应的训练 manifest。

    label_schema v2 起类别集合会变（新增 fruit，12 类），因此**必须**以该 checkpoint 记录的
    训练 manifest 为准，而不是全局常量，否则 category_idx→名称 会错位。
    返回 (manifest_dict, 相对路径或绝对路径, 校验说明)。
    """
    cfg = checkpoint.get('config') or {}
    rec = cfg.get('manifest')
    if rec:
        p = Path(rec)
        if not p.is_absolute():
            p = ROOT / rec
        if p.exists():
            if cfg.get('manifest_sha256') and file_digest(p) != cfg['manifest_sha256']:
                raise ValueError(f'五头权重记录的 manifest 已变化: {p}')
            try:
                shown = str(p.relative_to(ROOT))
            except ValueError:
                shown = str(p)
            if int(cfg.get('num_classes') or 0) and int(cfg['num_classes']) != len(
                    json.loads(p.read_text(encoding='utf-8'))['category_to_idx']):
                raise ValueError(f'五头权重记录的类别数与 manifest 不一致: {p}')
            return json.loads(p.read_text(encoding='utf-8')), shown, 'from_checkpoint_config'
    if not MACROS_MANIFEST.exists():
        raise FileNotFoundError(str(MACROS_MANIFEST))
    return read(MACROS_MANIFEST), str(MACROS_MANIFEST.relative_to(ROOT)), 'fallback_global_constant'


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
        self.macros = None
        self.macros_version = None
        self.macros_sha = None

    def load_macros_if_ready(self):
        """P1-B：加载五头宏量模型（若 checkpoint 存在），主输出切换到宏量模型。"""
        if self.macros is not None:
            return 'loaded'
        if not MACROS_CKPT.exists():
            return 'not_available'
        import hashlib as _hl
        ck = torch.load(MACROS_CKPT, map_location='cpu', weights_only=True)
        macros_manifest, shown, source = resolve_macros_manifest(ck)
        model = MealMacrosNet(macros_manifest, pretrained=False)
        model.load_state_dict(ck['model'], strict=True)
        model.to(self.device).eval()
        self.macros = model
        self.macros_version = MACROS_VERSION
        self.macros_sha = _hl.sha256(MACROS_CKPT.read_bytes()).hexdigest()
        # 类别映射与权重同步（12 类同理）：避免用旧 11 类映射解释 12 类 logits
        self.macros_manifest_path = shown
        self.macros_manifest_source = source
        self.macros_label_schema = macros_manifest.get('label_schema_version', 'v1_11class')
        self.macros_categories = {v: k for k, v in macros_manifest['category_to_idx'].items()}
        self.macros_status = 'supported'
        return 'loaded'

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
            # P1-B：主输出切换为五头宏量模型（热量/重量/类别/蛋白/碳水/脂肪），接受热量略于两目标
            self.load_macros_if_ready()
            if self.macros is not None:
                with torch.inference_mode():
                    mlogits, mvals = self.macros(tensor)
                if not torch.isfinite(mvals).all() or not torch.isfinite(mlogits).all():
                    raise FloatingPointError('Non-finite macros model output')
                result['calories'] = float(mvals[0, 0])
                result['weight'] = float(mvals[0, 1])
                result['protein_g'] = float(mvals[0, 2])
                result['carbohydrate_g'] = float(mvals[0, 3])
                result['fat_g'] = float(mvals[0, 4])
                result['macros_status'] = 'supported'
                mprob = mlogits.float().softmax(1)[0]
                mindex = int(mprob.argmax())
                # 类别名称用**五头权重自己的** manifest 映射（label_schema v2 为 12 类，含 fruit）
                mcat = self.macros_categories
                result['category_idx'] = mindex
                result['category_name'] = CATEGORY_ZH.get(mcat[mindex], mcat[mindex])
                result['category_prob'] = float(mprob[mindex])
                mtop = torch.argsort(mprob, descending=True)[:5].tolist()
                result['category_probs'] = [
                    {'idx': i, 'id': mcat[i], 'name': CATEGORY_ZH.get(mcat[i], mcat[i]),
                     'prob': round(float(mprob[i]), 3), 'pct': round(float(mprob[i]) * 100, 1)}
                    for i in mtop
                ]
                result['label_schema'] = self.macros_label_schema
                result['category_manifest'] = self.macros_manifest_path
                result['source_model'] = self.macros_version
                result['source_model_sha256'] = self.macros_sha
                result['target_names'] = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']
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
