"""
推理管线
=========
完整的端到端推理流程: 上传图 → NIR生成 → 分类 → 卡路里 → 食谱

流程:
    1. 输入RGB图像
    2. (可选) 食物分割 → 去背景
    3. NIR图像生成器 → 预测NIR
    4. 拼接RGB+NIR → 多任务网络 → {类别, 卡路里, 重量}
    5. (可选) 基线对比
    6. TDEE估算 + 食谱推荐

注意:
    - 多任务网络的类别数从 configs/default.yaml 读取, 必须与训练一致(16类)
    - 营养回归为log1p空间预测, 推理后需expm1还原到物理空间
    - 生成器输入为[-1,1]域的RGB(与train_multitask.py的make_4ch一致)
"""

import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, Any, List
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class InferencePipeline:
    """端到端推理管线

    Args:
        generator_ckpt: NIR生成器checkpoint路径 (None则不加载)
        multitask_ckpt: 多任务网络checkpoint路径 (None则不加载)
        config_path: 配置文件路径 (读取类别数)
        device: 计算设备
        use_segmentation: 是否使用食物分割预处理
        use_baseline: 是否使用CalorieCLIP基线对比
    """

    def __init__(
        self,
        generator_ckpt: Optional[str] = None,
        multitask_ckpt: Optional[str] = None,
        config_path: str = "configs/default.yaml",
        device: str = "auto",
        use_segmentation: bool = False,
        use_baseline: bool = False,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 从配置读取类别数, 与训练保持一致
        num_classes = None
        cfg_path = config_path if os.path.exists(config_path) else os.path.join(PROJECT_ROOT, config_path)
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                num_classes = cfg.get('multitask', {}).get('num_classes')
            except Exception as e:
                print(f"配置读取警告: {e}")
        if not num_classes:
            from data.nutrition5k_loader import CATEGORY_CLASSES
            num_classes = len(CATEGORY_CLASSES)
        self.num_classes = num_classes

        # 类别名称表
        try:
            from data.nutrition5k_loader import CATEGORY_CLASSES
            self.category_names = list(CATEGORY_CLASSES)
        except Exception:
            self.category_names = None

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.transform_simple = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # 加载模型
        self.generator = None
        self.multitask = None
        self.segmentation = None
        self.baseline = None

        if generator_ckpt and os.path.exists(generator_ckpt):
            self._load_generator(generator_ckpt)
        else:
            print(f"警告: NIR生成器未加载 ({generator_ckpt}), 将使用零通道NIR")

        if multitask_ckpt and os.path.exists(multitask_ckpt):
            self._load_multitask(multitask_ckpt)
        else:
            print(f"警告: 多任务网络未加载 ({multitask_ckpt}), 无法预测")

        if use_segmentation:
            self._load_segmentation()

        if use_baseline:
            self._load_baseline()

    def _load_generator(self, ckpt_path: str):
        """加载NIR生成器 (键处理与evaluate_multitask.py一致)"""
        from models.generator.unet import UNetGenerator
        self.generator = UNetGenerator(input_channels=3, output_channels=1).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)

        if any(k.startswith('generator.') for k in state):
            # Pix2Pix训练checkpoint: generator./discriminator.前缀
            state = {k.replace('generator.', '', 1): v
                     for k, v in state.items() if k.startswith('generator.')}
        else:
            state = {k.replace('module.', ''): v
                     for k, v in state.items() if 'discriminator' not in k}

        incompatible = self.generator.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            print(f"生成器加载警告: {len(incompatible.missing_keys)}个参数缺失, "
                  f"示例: {incompatible.missing_keys[:3]}")
        self.generator.eval()
        print(f"NIR生成器已加载: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

    def _load_multitask(self, ckpt_path: str):
        """加载多任务网络 (类别数必须与训练一致)"""
        from models.multitask.resnet_multitask import ResNetMultiTask
        self.multitask = ResNetMultiTask(
            num_classes=self.num_classes, input_channels=4, pretrained=False,
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        # strict=True: 立刻暴露类别数/通道数不匹配, 避免静默使用随机权重
        self.multitask.load_state_dict(state)
        self.multitask.eval()
        print(f"多任务网络已加载: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

    def _load_segmentation(self):
        """加载食物分割模型"""
        from optimizations.food_segmentation import FoodSegmentationPreprocessor
        self.segmentation = FoodSegmentationPreprocessor(freeze_segmentation=True).to(self.device)
        self.segmentation.eval()
        print("食物分割模型已加载(未训练)")

    def _load_baseline(self):
        """加载CalorieCLIP基线"""
        try:
            from models.baseline.calorieclip_wrapper import CalorieCLIPWrapper
            self.baseline = CalorieCLIPWrapper(freeze_encoder=True).to(self.device)
            self.baseline.init_classifier(num_classes=self.num_classes)
            self.baseline.eval()
            print("CalorieCLIP基线已加载")
        except Exception as e:
            print(f"CalorieCLIP加载失败: {e}")
            self.baseline = None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """完整推理流程

        Args:
            image: 输入食物RGB图像 (PIL Image)

        Returns:
            dict: {
                'original_image':  PIL Image
                'nir_image':       numpy array (NIR预测, 0-255灰度)
                'segmentation_mask': numpy array (如果启用分割)
                'category':        str (食物类别名称)
                'category_idx':    int (类别索引)
                'category_prob':   float (分类置信度)
                'calories':        float (预测卡路里 kcal, 已expm1还原)
                'weight':          float (预测重量 g, 已expm1还原)
                'baseline_calories': float (基线卡路里, 可选)
            }
        """
        result = {"original_image": image}

        # 预处理
        rgb_tensor = self.transform(image).unsqueeze(0).to(self.device)    # ImageNet归一化, 多任务网络输入
        rgb_simple = self.transform_simple(image).unsqueeze(0).to(self.device)  # [0,1]域

        # 食物分割(可选)
        if self.segmentation is not None:
            with torch.no_grad():
                seg_out = self.segmentation(rgb_simple)
            mask = seg_out["mask"][0, 0].cpu().numpy()
            result["segmentation_mask"] = mask
            rgb_tensor = seg_out["masked_image"]

        # NIR生成: 输入需转到[-1,1]域, 与train_multitask.py的make_4ch一致
        if self.generator is not None:
            with torch.no_grad():
                nir_tensor = self.generator(rgb_simple * 2.0 - 1.0)
        else:
            nir_tensor = torch.zeros(1, 1, 256, 256).to(self.device)

        # NIR图像输出 (生成器输出[-1,1], 转成0-255用于显示)
        nir_image = nir_tensor[0, 0].cpu().numpy()
        nir_image = ((nir_image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        result["nir_image"] = nir_image

        # 多任务推理
        if self.multitask is not None:
            # 拼接RGB+NIR (与训练一致: ImageNet归一化RGB + [-1,1]NIR)
            input_4ch = torch.cat([rgb_tensor, nir_tensor], dim=1)
            with torch.no_grad():
                outputs = self.multitask(input_4ch)

            # 分类
            probs = torch.softmax(outputs['logits'], dim=1)
            top_prob, top_idx = probs[0].max(dim=0)

            idx = top_idx.item()
            result["category_idx"] = idx
            if self.category_names and idx < len(self.category_names):
                result["category"] = self.category_names[idx]
            else:
                result["category"] = f"类别{idx}"
            result["category_prob"] = top_prob.item()

            # 回归: log1p空间 → expm1还原物理空间
            nutrition = outputs['nutrition'][0].cpu().numpy()
            result["calories"] = max(0.0, float(np.expm1(nutrition[0])))
            result["weight"] = max(0.0, float(np.expm1(nutrition[1])))
        else:
            result["category"] = "未加载模型"
            result["category_idx"] = -1
            result["category_prob"] = 0.0
            result["calories"] = 0.0
            result["weight"] = 0.0

        # 基线对比(可选)
        if self.baseline is not None:
            try:
                with torch.no_grad():
                    baseline_out = self.baseline(rgb_tensor)
                result["baseline_calories"] = float(baseline_out['nutrition'][0, 0].cpu().numpy())
            except Exception:
                result["baseline_calories"] = None

        return result


if __name__ == "__main__":
    pipeline = InferencePipeline(
        generator_ckpt="./checkpoints/generator/best_model.pt",
        multitask_ckpt="./checkpoints/multitask/best_model.pt",
        device="cpu",
    )

    # 测试随机输入
    from PIL import Image as PILImage
    import numpy as np
    test_img = PILImage.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    result = pipeline.predict(test_img)
    print(f"预测结果: {result.get('category', 'N/A')}, "
          f"{result.get('calories', 0):.0f} kcal, {result.get('weight', 0):.0f} g")
