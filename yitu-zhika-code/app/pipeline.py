"""
推理管线 (本地适配版)
=====================
端到端推理: 上传RGB图 → NIR生成 → 4通道拼接 → 分类+卡路里回归

支持两种生成器架构:
    - 7层 Deep U-Net (nir_generator, 层名 down1-7) — 暂无checkpoint
    - 4层 U-Net (generator, 层名 enc1-4) — 有200轮checkpoint
    根据checkpoint的state_dict键名自动选择。

使用方法:
    from app.pipeline import InferencePipeline
    pipe = InferencePipeline(
        generator_ckpt="checkpoints/phase1/final_model.pth",
        multitask_ckpt="checkpoints/multitask/best_model.pt",
    )
    result = pipe.predict(pil_image)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, Any
from torchvision import transforms

# 路径设置: 同时添加项目根目录和src目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)


class InferencePipeline:
    """端到端推理管线

    Args:
        generator_ckpt: NIR生成器checkpoint路径
        multitask_ckpt: 多任务网络checkpoint路径
        device: "auto" | "cuda" | "cpu"
    """

    def __init__(
        self,
        generator_ckpt: Optional[str] = None,
        multitask_ckpt: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # RGB预处理: ImageNet归一化 (用于多任务网络输入)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # RGB预处理: 仅ToTensor [0,1] (用于生成器输入, 会手动转[-1,1])
        self.transform_simple = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.generator = None
        self.multitask = None

        for path in (generator_ckpt, multitask_ckpt):
            if path is not None and not os.path.isfile(path):
                raise FileNotFoundError(f'Checkpoint not found: {path}')

        if generator_ckpt and os.path.exists(generator_ckpt):
            self._load_generator(generator_ckpt)
        else:
            print("[WARN] NIR生成器未加载，将使用零通道NIR")

        if multitask_ckpt and os.path.exists(multitask_ckpt):
            self._load_multitask(multitask_ckpt)
        else:
            print("[WARN] 多任务网络未加载")

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
            state = ckpt

        # 过滤discriminator键
        state = {k.replace('module.', ''): v for k, v in state.items()
                 if 'discriminator' not in k.lower() and 'D_state' not in k}

        # 根据state_dict键名判断架构
        has_down5 = any(k.startswith('down5') for k in state.keys())
        has_enc1 = any(k.startswith('enc1') for k in state.keys())

        if has_down5:
            from models.nir_generator import UNetGenerator
            self.generator = UNetGenerator(
                in_channels=3, out_channels=1, base_channels=64
            ).to(self.device)
            arch_name = "7-layer Deep U-Net (nir_generator)"
        elif has_enc1:
            from models.generator import UNetGenerator
            self.generator = UNetGenerator(
                in_channels=3, out_channels=1, base_filters=64
            ).to(self.device)
            arch_name = "4-layer U-Net (generator)"
        else:
            raise ValueError('Unknown generator architecture; refusing partial inference')

        result = self.generator.load_state_dict(state, strict=True)
        if result.missing_keys:
            print(f"  [WARN] Missing keys: {len(result.missing_keys)} layers")
        if result.unexpected_keys:
            print(f"  [WARN] Unexpected keys: {len(result.unexpected_keys)} layers")
        if not result.missing_keys and not result.unexpected_keys:
            print(f"  [OK] All keys matched")

        self.generator.eval()
        epoch = ckpt.get('epoch', '?')
        print(f"[OK] NIR generator loaded [{arch_name}] (epoch {epoch})")

    def _load_multitask(self, ckpt_path: str):
        """加载多任务网络"""
        from models.checkpoint_io import load_multitask_checkpoint
        self.multitask, self.multitask_metadata = load_multitask_checkpoint(ckpt_path, self.device)
        print(f"[OK] Strict multitask load; targets={self.multitask_metadata['target_names']}")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """完整推理流程

        Args:
            image: 输入食物RGB图像 (PIL Image)

        Returns:
            dict with keys: nir_image, category_idx, category_prob, calories, weight
        """
        if self.multitask is None:
            raise ValueError('Inference requires a loaded multitask checkpoint; no placeholder estimates')
        image = image.convert('RGB')
        result = {}

        # 预处理
        rgb_tensor = self.transform(image).unsqueeze(0).to(self.device)
        rgb_simple = self.transform_simple(image).unsqueeze(0).to(self.device)

        # NIR生成
        if self.generator is not None:
            with torch.no_grad():
                rgb_for_gen = rgb_simple * 2.0 - 1.0
                nir_tensor = self.generator(rgb_for_gen)
        else:
            nir_tensor = torch.zeros(1, 1, 256, 256).to(self.device)

        # NIR图像输出
        nir_image = nir_tensor[0, 0].cpu().numpy()
        nir_image = ((nir_image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        result["nir_image"] = nir_image

        # 多任务推理
        if self.multitask is not None:
            nir_01 = (nir_tensor + 1.0) / 2.0
            nir_imagenet = (nir_01 - 0.485) / 0.229
            if self.multitask.input_channels == 4:
                if self.generator is None:
                    raise ValueError('RGB+NIR inference requires a loaded generator')
                model_input = torch.cat([rgb_tensor, nir_imagenet], dim=1)
            elif self.multitask.input_channels == 3:
                model_input = rgb_tensor
            else:
                raise ValueError('Unsupported checkpoint input channels')

            with torch.no_grad():
                outputs = self.multitask(model_input)

            probs = torch.softmax(outputs['logits'], dim=1)
            top_prob, top_idx = probs[0].max(dim=0)
            valid_cls = self.multitask_metadata['classification_valid']
            result["category_idx"] = top_idx.item() if valid_cls else -1
            result["category_prob"] = top_prob.item() if valid_cls else 0.0
            result["classification_valid"] = valid_cls

            nutrition = outputs['nutrition'][0].cpu().numpy()
            from models.checkpoint_io import named_nutrition
            values = named_nutrition(nutrition, self.multitask_metadata['target_names'])
            result["calories"] = values['calories']
            result["weight"] = values['mass']
            for name in ('protein', 'carb', 'fat'):
                if name in values:
                    result[name] = values[name]
            result['model_notice'] = self.multitask_metadata.get('limitations', '')
        else:
            result["category_idx"] = -1
            result["category_prob"] = 0.0
            result["calories"] = 0.0
            result["weight"] = 0.0

        return result


if __name__ == "__main__":
    pipeline = InferencePipeline(device="cpu")
    print("Pipeline initialized (untrained models)")

    test_img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    result = pipeline.predict(test_img)
    print(f"Predicted: {result.get('calories', 'N/A')} kcal")
