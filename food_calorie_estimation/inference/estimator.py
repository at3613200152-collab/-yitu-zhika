"""
端到端推理器
============
把 NIR 生成器 + 多任务回归网络 + (可选) 纯 RGB 基线 串成一条推理管线，
并提供统一的 predict_all() 给前端 /server.py 调用。

加载策略:
- 如果提供了 multitask_ckpt, 用训练好的多任务网络做真实预测
- 如果没提供 / 加载失败, 自动 fallback 到 ImageNet 预训练的 ResNet50 + 知识库兜底
- baseline 始终跑纯 RGB 路径 (use_nir=False)
"""

import os
import io
import base64
import random
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms, models

from models.nir_generator import UNetGenerator
from models.multitask_net import MultiTaskResNet
from utils.utils import load_config, tensor_to_image

from .class_mapping import (
    NUTRITION5K_CLASSES,
    NUM_CLASSES,
    get_class_info,
    get_calories_per_100g,
)
from .food_knowledge import FOOD_KNOWLEDGE_BASE, get_food_list


class FoodCalorieEstimator:
    """完整的卡路里估计推理器"""

    def __init__(
        self,
        config_path: str = "config.yaml",
        nir_ckpt: Optional[str] = None,
        multitask_ckpt: Optional[str] = None,
        baseline_ckpt: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Estimator] device = {self.device}")

        # 模型加载
        self.nir_generator = self._load_nir_generator(nir_ckpt)
        multitask_loaded, self.multitask = self._load_multitask(multitask_ckpt)
        self.multitask_loaded = multitask_loaded
        self.imagenet_fallback = self._load_imagenet_fallback()
        self._log_target = self.config["data"]["nutrition5k"].get("log_target", True)

        # 图像变换
        image_size = self.config["data"]["hsifoodingr64"]["image_size"]
        self.nir_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.multitask_rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.classify_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------ 模型加载
    def _load_nir_generator(self, ckpt_path: Optional[str]):
        cfg = self.config["models"]["nir_generator"]
        model = UNetGenerator(
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            base_channels=cfg["base_channels"],
            num_downs=cfg["num_downs"],
        ).to(self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state = (
                ckpt.get("generator_state_dict")
                or ckpt.get("model_state_dict")
                or ckpt
            )
            try:
                model.load_state_dict(state)
                print(f"[Estimator] NIR generator loaded: {ckpt_path}")
            except Exception as e:
                print(f"[Estimator] NIR generator load warning: {e}")
        else:
            print("[Estimator] No NIR generator checkpoint found, using random init.")
        model.eval()
        return model

    def _load_multitask(self, ckpt_path: Optional[str]) -> Tuple[bool, Optional[MultiTaskResNet]]:
        if not (ckpt_path and os.path.exists(ckpt_path)):
            print("[Estimator] No multitask checkpoint found.")
            return False, None
        cfg = self.config["models"]["multitask"]
        model = MultiTaskResNet(self.config).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state)
            print(f"[Estimator] Multitask model loaded: {ckpt_path}")
        except Exception as e:
            print(f"[Estimator] Multitask load warning: {e}")
            return False, None
        model.eval()
        return True, model

    def _load_imagenet_fallback(self):
        """用 ImageNet ResNet50 + 知识库做兜底, 即便训练任务未完成也能 demo"""
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
            model.eval().to(self.device)
            self.imagenet_categories = weights.meta["categories"]
            print(f"[Estimator] ImageNet ResNet50 fallback ready "
                  f"({len(self.imagenet_categories)} classes).")
            return model
        except Exception as e:
            print(f"[Estimator] ImageNet fallback unavailable: {e}")
            return None

    # ------------------------------------------------------------------ 图像预处理
    @staticmethod
    def _open_image(rgb_image) -> Image.Image:
        if isinstance(rgb_image, Image.Image):
            return rgb_image.convert("RGB")
        if isinstance(rgb_image, (bytes, bytearray)):
            return Image.open(io.BytesIO(rgb_image)).convert("RGB")
        if isinstance(rgb_image, str):
            return Image.open(rgb_image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(rgb_image)}")

    # ------------------------------------------------------------------ NIR 生成
    @torch.no_grad()
    def predict_nir(self, rgb_image) -> Tuple[Image.Image, torch.Tensor]:
        img = self._open_image(rgb_image)
        x = self.nir_transform(img).unsqueeze(0).to(self.device)
        nir = self.nir_generator(x)
        nir_img = tensor_to_image(nir, normalize=True)
        if nir_img.shape[2] == 1:
            nir_img = nir_img[:, :, 0]
        return Image.fromarray(nir_img), nir.squeeze(0)

    # ------------------------------------------------------------------ 分类 + 回归
    @torch.no_grad()
    def _predict_multitask_tensor(
        self, rgb_image: Image.Image, nir_tensor: Optional[torch.Tensor] = None
    ) -> Dict:
        """用训练好的多任务网络做预测.

        Returns:
            dict: {
                'top5': [...],          # Top5 类别 + 概率
                'calories': float,      # 预测卡路里
                'weight': float,        # 预测重量 (g)
                'used_nir': bool,       # 是否用 NIR
            }
        """
        if self.multitask is None:
            raise RuntimeError("multitask model not loaded")

        rgb = self.multitask_rgb_transform(rgb_image)
        if nir_tensor is not None:
            # 将 NIR 调整到与 RGB 同样的空间尺寸
            nir_resized = F.interpolate(
                nir_tensor.unsqueeze(0),
                size=rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            # 多任务网络 in_channels=4
            x = torch.cat([rgb, nir_resized[:1]], dim=0).unsqueeze(0).to(self.device)
        else:
            # 纯 RGB: 用 0 填充 NIR 通道
            zero_nir = torch.zeros(1, rgb.shape[-2], rgb.shape[-1])
            x = torch.cat([rgb, zero_nir], dim=0).unsqueeze(0).to(self.device)

        outputs = self.multitask(x.to(self.device))
        logits = outputs["cls_logits"]
        probs = F.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.topk(probs, k=min(5, probs.shape[0]))
        top_idx_list = top_idx.cpu().tolist()
        top_prob_list = top_prob.cpu().tolist()

        top5 = []
        for cid, p in zip(top_idx_list, top_prob_list):
            name_en, name_cn, category, cal100 = get_class_info(cid)
            top5.append({
                "class": cid,
                "key": name_en,
                "name": name_cn,
                "name_en": name_en,
                "probability": float(p),
                "category": category,
                "calories_per_100g": cal100,
            })

        cal = float(outputs["calories"].item())
        weight = float(outputs["weight"].item())
        # 模型直接回归结果可能是负数，做合理裁剪
        cal = max(0.0, cal)
        weight = max(0.0, weight)

        # 如果训练时用了 log1p, 这里把预测还原
        if self._log_target:
            cal = float(np.expm1(cal))
            weight = float(np.expm1(weight))

        return {
            "top5": top5,
            "calories": cal,
            "weight": weight,
            "used_nir": nir_tensor is not None,
        }

    # ------------------------------------------------------------------ 兜底: ImageNet + 知识库
    def _predict_fallback(self, rgb_image: Image.Image) -> Dict:
        if self.imagenet_fallback is None:
            return self._predict_color_heuristic(rgb_image)

        x = self.classify_transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.imagenet_fallback(x)
            probs = F.softmax(logits, dim=1).squeeze(0)

        knowledge = FOOD_KNOWLEDGE_BASE
        food_keys = get_food_list()
        # 对每个知识库食物做匹配
        scores = {k: 0.0 for k in food_keys}
        for food_key in food_keys:
            info = knowledge[food_key]
            food_name = food_key.replace("_", " ")
            for i, inet_class in enumerate(self.imagenet_categories):
                inet_lower = inet_class.lower().replace("_", " ")
                if food_name in inet_lower or inet_lower in food_name:
                    scores[food_key] += float(probs[i].item())
                for prompt in info.get("prompts", []):
                    pc = prompt.replace("_", " ").lower()
                    if pc in inet_lower or inet_lower in pc:
                        scores[food_key] += float(probs[i].item())

        total = sum(scores.values()) or 1.0
        probs_norm = {k: v / total for k, v in scores.items()}
        sorted_items = sorted(probs_norm.items(), key=lambda x: x[1], reverse=True)[:5]

        top5 = []
        for key, p in sorted_items:
            info = knowledge[key]
            top5.append({
                "class": food_keys.index(key) if key in food_keys else -1,
                "key": key,
                "name": info["name_cn"],
                "name_en": key,
                "probability": float(p),
                "category": info["category"],
                "calories_per_100g": info["calories_per_100g"],
            })

        best = top5[0]
        # 用知识库的典型值做兜底
        base_cal = knowledge[best["key"]]["typical_calories"]
        base_weight = knowledge[best["key"]]["typical_weight"]
        rng = random.Random(hash(best["key"]) % 2**31)
        cal = base_cal * (1.0 + (rng.random() - 0.5) * 0.15)
        weight = base_weight * (1.0 + (rng.random() - 0.5) * 0.12)

        return {
            "top5": top5,
            "calories": round(cal, 1),
            "weight": round(weight, 1),
            "used_nir": False,
        }

    def _predict_color_heuristic(self, rgb_image: Image.Image) -> Dict:
        """最终兜底: 用颜色直方图做粗略分类"""
        arr = np.array(rgb_image.resize((64, 64))).astype(np.float32) / 255.0
        avg_r, avg_g, avg_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        brightness = (avg_r + avg_g + avg_b) / 3.0
        sat = max(avg_r, avg_g, avg_b) - min(avg_r, avg_g, avg_b)
        rg_ratio = avg_r / max(avg_g, 0.01)

        knowledge = FOOD_KNOWLEDGE_BASE
        food_keys = get_food_list()
        scored: List[Tuple[str, float]] = []
        for key in food_keys:
            info = knowledge[key]
            score = 0.01
            cat = info["category"]
            if cat == "水果":
                score += 0.18 if rg_ratio > 1.2 else 0.05
                score += 0.08 * sat
            elif cat == "蔬菜":
                if avg_g > avg_r and avg_g > avg_b:
                    score += 0.2
                score += 0.05 * sat
            elif cat == "肉类":
                if avg_r > avg_g and avg_r > avg_b:
                    score += 0.18
                score += 0.04 * (1 - brightness)
            elif cat == "主食":
                score += 0.15 if brightness > 0.4 else 0
                score += 0.04 * max(0, brightness - 0.3)
            elif cat == "甜品":
                score += 0.12 if brightness > 0.45 else 0
                score += 0.05 * brightness
            elif cat == "快餐":
                score += 0.15 if brightness > 0.35 and sat > 0.2 else 0
                score += 0.04 * sat
            elif cat == "海鲜":
                score += 0.1 if avg_r < avg_g + 0.1 and brightness < 0.5 else 0
            elif cat == "乳制品":
                score += 0.16 if brightness > 0.5 else 0
            scored.append((key, max(score, 0.01)))

        scored.sort(key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in scored) or 1.0
        top5 = []
        for key, s in scored[:5]:
            info = knowledge[key]
            top5.append({
                "class": food_keys.index(key) if key in food_keys else -1,
                "key": key,
                "name": info["name_cn"],
                "name_en": key,
                "probability": float(s / total),
                "category": info["category"],
                "calories_per_100g": info["calories_per_100g"],
            })
        best = top5[0]
        info = knowledge[best["key"]]
        return {
            "top5": top5,
            "calories": info["typical_calories"],
            "weight": info["typical_weight"],
            "used_nir": False,
        }

    # ------------------------------------------------------------------ 统一接口
    def predict_multitask(self, rgb_image, use_nir: bool = True) -> Dict:
        """统一的多任务预测接口 (前向 + 兜底)"""
        img = self._open_image(rgb_image)
        nir_tensor = None
        if use_nir:
            _, nir_tensor = self.predict_nir(img)

        if self.multitask is not None:
            try:
                res = self._predict_multitask_tensor(img, nir_tensor=nir_tensor)
                res["source"] = "multitask_model"
                return res
            except Exception as e:
                print(f"[Estimator] multitask predict failed: {e}")
        # 兜底
        res = self._predict_fallback(img)
        res["source"] = "imagenet_fallback"
        return res

    def predict_all(self, rgb_image) -> Dict:
        """前端 /api/predict 主入口, 返回:
        - rgb_image, nir_image (base64)
        - multispectral: {food_class, calories, weight, ...}
        - baseline:      {food_class, calories, weight, ...}     # 纯 RGB
        - comparison:    两种方法的差异
        """
        img = self._open_image(rgb_image)
        nir_pil, nir_tensor = self.predict_nir(img)

        # base64
        def _to_b64(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        rgb_b64 = _to_b64(img)
        nir_b64 = _to_b64(nir_pil)

        ms = self.predict_multitask(img, use_nir=True)
        bl = self.predict_multitask(img, use_nir=False)

        ms_top = ms["top5"][0]
        bl_top = bl["top5"][0]

        cal_diff = ms["calories"] - bl["calories"]
        rel_change = abs(cal_diff) / max(abs(bl["calories"]), 1e-6) * 100
        return {
            "rgb_image": rgb_b64,
            "nir_image": nir_b64,
            "multispectral": {
                "method": "多光谱 (RGB + 预测 NIR)",
                "food_class": ms_top["name"],
                "food_class_en": ms_top["name_en"],
                "class_probability": ms_top["probability"],
                "category": ms_top["category"],
                "calories_per_100g": ms_top["calories_per_100g"],
                "calories": round(ms["calories"], 1),
                "weight": round(ms["weight"], 1),
                "top5_classes": ms["top5"],
                "source": ms.get("source", "unknown"),
            },
            "baseline": {
                "method": "基线 (纯 RGB)",
                "food_class": bl_top["name"],
                "food_class_en": bl_top["name_en"],
                "class_probability": bl_top["probability"],
                "category": bl_top["category"],
                "calories_per_100g": bl_top["calories_per_100g"],
                "calories": round(bl["calories"], 1),
                "weight": round(bl["weight"], 1),
                "top5_classes": bl["top5"],
                "source": bl.get("source", "unknown"),
            },
            "comparison": {
                "calories_diff": round(cal_diff, 1),
                "relative_change_pct": round(rel_change, 2),
                "abs_mean_diff": round(abs(cal_diff), 1),
            },
            "model_status": {
                "nir_generator_loaded": self.nir_generator is not None,
                "multitask_loaded": self.multitask is not None,
                "imagenet_fallback_loaded": self.imagenet_fallback is not None,
            },
        }

    # ------------------------------------------------------------------ 实用方法
    def get_model_status(self) -> Dict:
        return {
            "nir_generator_loaded": self.nir_generator is not None,
            "multitask_loaded": self.multitask is not None,
            "imagenet_fallback_loaded": self.imagenet_fallback is not None,
            "device": str(self.device),
            "num_classes": NUM_CLASSES,
        }
