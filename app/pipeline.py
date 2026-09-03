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

支持两种生成器架构:
    - nir_generator (7层Deep U-Net, 层名 down1-7/up0-7)
    - generator_v1 (4层U-Net, 层名 enc1-4/dec1-4)
    根据checkpoint的state_dict键自动选择。
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Optional, Any
from torchvision import transforms

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class InferencePipeline:
    """端到端推理管线

    Args:
        generator_ckpt: NIR生成器checkpoint路径
        multitask_ckpt: 多任务网络checkpoint路径
        device: 计算设备
        use_segmentation: 是否使用食物分割预处理
        use_baseline: 是否使用CalorieCLIP基线对比
    """

    def __init__(
        self,
        generator_ckpt: Optional[str] = None,
        multitask_ckpt: Optional[str] = None,
        device: str = "auto",
        use_segmentation: bool = False,
        use_baseline: bool = False,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
            print("⚠️ NIR生成器未加载，将使用零通道NIR")

        if multitask_ckpt and os.path.exists(multitask_ckpt):
            self._load_multitask(multitask_ckpt)

        if use_segmentation:
            self._load_segmentation()

        if use_baseline:
            self._load_baseline()

    def _load_generator(self, ckpt_path: str):
        """加载NIR生成器 — 自动检测架构"""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # 提取generator state_dict (兼容多种key格式)
        state = None
        for key in ['G_state_dict', 'generator_state_dict', 'model_state_dict']:
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            # 直接用整个ckpt（假设就是state_dict）
            state = ckpt

        # 过滤掉discriminator的键
        state = {k.replace('module.', ''): v for k, v in state.items()
                 if 'discriminator' not in k.lower() and 'D_state' not in k}

        # 根据state_dict的键名判断架构
        has_down5 = any(k.startswith('down5') for k in state.keys())
        has_enc1 = any(k.startswith('enc1') for k in state.keys())

        if has_down5:
            # 7层 Deep U-Net (nir_generator)
            from models.generator.nir_generator import UNetGenerator
            self.generator = UNetGenerator(input_channels=3, output_channels=1).to(self.device)
            arch_name = "7层Deep U-Net (nir_generator)"
        elif has_enc1:
            # 4层 U-Net (generator_v1)
            from models.generator.generator_v1 import UNetGenerator
            self.generator = UNetGenerator(input_channels=3, out_channels=1).to(self.device)
            arch_name = "4层U-Net (generator_v1)"
        else:
            # 默认尝试团队的unet.py
            try:
                from models.generator.unet import UNetGenerator
                self.generator = UNetGenerator(input_channels=3, output_channels=1).to(self.device)
                arch_name = "团队UNet (unet.py)"
            except ImportError:
                print("❌ 无法确定生成器架构，跳过加载")
                return

        try:
            self.generator.load_state_dict(state, strict=False)
            missing, unexpected = self.generator.load_state_dict(state, strict=False)
            if missing:
                print(f"  ⚠️ Missing keys: {len(missing)} keys")
            if unexpected:
                print(f"  ⚠️ Unexpected keys: {len(unexpected)} keys")
        except Exception as e:
            print(f"  ⚠️ 生成器加载警告: {e}")

        self.generator.eval()
        epoch = ckpt.get('epoch', '?')
        print(f"✅ NIR生成器已加载 [{arch_name}] (epoch {epoch})")

    def _load_multitask(self, ckpt_path: str):
        """加载多任务网络"""
        from models.multitask.resnet_multitask import ResNetMultiTask
        self.multitask = ResNetMultiTask(
            num_classes=61, input_channels=4, pretrained=False,
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('net_state_dict', ckpt))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        try:
            self.multitask.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"  ⚠️ 多任务网络加载警告: {e}")
        self.multitask.eval()
        print(f"✅ 多任务网络已加载")

    def _load_segmentation(self):
        """加载食物分割模型"""
        try:
            from optimizations.food_segmentation import FoodSegmentationPreprocessor
            self.segmentation = FoodSegmentationPreprocessor(freeze_segmentation=True).to(self.device)
            self.segmentation.eval()
            print("  食物分割模型已加载(未训练)")
        except Exception as e:
            print(f"  食物分割加载失败: {e}")
            self.segmentation = None

    def _load_baseline(self):
        """加载CalorieCLIP基线"""
        try:
            from models.baseline.calorieclip_wrapper import CalorieCLIPWrapper
            self.baseline = CalorieCLIPWrapper(freeze_encoder=True).to(self.device)
            self.baseline.init_classifier(num_classes=61)
            self.baseline.eval()
            print("  CalorieCLIP基线已加载")
        except Exception as e:
            print(f"  CalorieCLIP加载失败: {e}")
            self.baseline = None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """完整推理流程

        Args:
            image: 输入食物RGB图像 (PIL Image)

        Returns:
            dict: {
                'original_image':  PIL Image
                'nir_image':       numpy array (NIR预测)
                'segmentation_mask': numpy array (如果启用分割)
                'category_idx':    int (食物类别索引)
                'category_prob':   float (分类置信度)
                'calories':        float (预测卡路里 kcal)
                'weight':          float (预测重量 g)
                'baseline_calories': float (基线卡路里, 可选)
            }
        """
        result = {"original_image": image}

        # 预处理
        rgb_tensor = self.transform(image).unsqueeze(0).to(self.device)
        rgb_simple = self.transform_simple(image).unsqueeze(0).to(self.device)

        # 食物分割(可选)
        if self.segmentation is not None:
            with torch.no_grad():
                seg_out = self.segmentation(rgb_simple)
            mask = seg_out["mask"][0, 0].cpu().numpy()
            result["segmentation_mask"] = mask
            rgb_tensor = seg_out["masked_image"]

        # NIR生成
        if self.generator is not None:
            with torch.no_grad():
                nir_tensor = self.generator(rgb_simple)
        else:
            nir_tensor = torch.zeros(1, 1, 256, 256).to(self.device)

        # NIR图像输出 (值域[-1,1] → [0,255])
        nir_image = nir_tensor[0, 0].cpu().numpy()
        nir_image = ((nir_image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        result["nir_image"] = nir_image

        # 多任务推理
        if self.multitask is not None:
            # RGB: ImageNet归一化, NIR: [-1,1]→ImageNet归一化
            nir_norm = transforms.Normalize(mean=[0.485], std=[0.229])
            nir_normalized = nir_norm(nir_tensor)
            input_4ch = torch.cat([rgb_tensor, nir_normalized], dim=1)

            with torch.no_grad():
                outputs = self.multitask(input_4ch)

            # 分类
            probs = torch.softmax(outputs['logits'], dim=1)
            top_prob, top_idx = probs[0].max(dim=0)

            result["category_idx"] = top_idx.item()
            result["category_prob"] = top_prob.item()

            # 回归
            nutrition = outputs['nutrition'][0].cpu().numpy()
            result["calories"] = float(nutrition[0])
            result["weight"] = float(nutrition[1])
        else:
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
    pipeline = InferencePipeline(device="cpu")
    print("推理管线初始化完成（模型未训练）")

    # 测试随机输入
    from PIL import Image as PILImage
    test_img = PILImage.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    result = pipeline.predict(test_img)
    print(f"预测结果: {result.get('calories', 'N/A')} kcal")
