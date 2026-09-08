"""CalorieCLIP 基线封装（吸收自队友 pure-pix2pix 分支）

来源: 队友 pure-pix2pix 分支 `models/baseline/calorieclip_wrapper.py`
适配: 仅调整 docstring 引用路径，原代码逻辑保持不变

CalorieCLIP 利用 CLIP 的视觉-语言对齐能力估计食物热量。
论文: Mini et al., "CalorieCLIP: Calorie Estimation from Food Images
      using Vision-Language Models"

用途: 作为我们主多任务网络 (ResNet50 + 5维回归) 的对比基线
      用法: from models.baseline.calorieclip_wrapper import CalorieCLIPWrapper
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image


class CalorieCLIPWrapper(nn.Module):
    """CalorieCLIP 基线模型封装

    使用 OpenCLIP 的预训练视觉编码器提取食物图像特征，
    然后通过回归头预测卡路里。

    本封装提供两种模式:
    1. 零样本模式: 直接使用 CLIP 的视觉-文本对齐
    2. 微调模式: 冻结 CLIP 视觉编码器，只训练回归头

    Args:
        model_name: OpenCLIP 模型名称 (默认 'ViT-B-32')
        pretrained: 预训练权重标识 (默认 'laion2b_s34b_b79k')
        freeze_encoder: 是否冻结视觉编码器 (默认 True)
        num_regression_outputs: 回归输出维度 (默认 2: 卡路里+重量)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        freeze_encoder: bool = True,
        num_regression_outputs: int = 2,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder

        try:
            import open_clip
            # 创建 CLIP 模型
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            # 获取 tokenize 函数
            self.tokenize = open_clip.tokenize

            # 获取视觉编码器的输出维度
            if hasattr(self.clip_model, 'visual'):
                visual = self.clip_model.visual
                if hasattr(visual, 'output_dim'):
                    self.visual_dim = visual.output_dim
                elif hasattr(visual, 'proj'):
                    self.visual_dim = visual.proj.shape[1]
                else:
                    # 默认 ViT-B-32 输出维度
                    self.visual_dim = 512
            else:
                self.visual_dim = 512

        except ImportError:
            print("警告: open_clip 未安装，CalorieCLIP 功能不可用")
            print("请运行: pip install open-clip-torch")
            self.clip_model = None
            self.visual_dim = 512

        # 冻结视觉编码器
        if freeze_encoder and self.clip_model is not None:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 回归头: CLIP 特征 → 营养值
        self.regressor = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_regression_outputs),
        )

        # 分类头: CLIP 特征 → 食物类别
        self.classifier = None  # 延迟初始化（需要知道 num_classes）

    def init_classifier(self, num_classes: int):
        """初始化分类头（需要知道类别数）"""
        self.classifier = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """提取图像特征

        Args:
            image: 输入图像 [B, 3, H, W]

        Returns:
            图像特征 [B, visual_dim]
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP 模型未加载")

        features = self.clip_model.encode_image(image)
        # 归一化特征
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            image: 输入图像 [B, 3, H, W]

        Returns:
            dict: {
                'features':  [B, visual_dim]  CLIP 视觉特征
                'nutrition': [B, 2]           回归输出
                'logits':    [B, num_classes]  分类输出(如果初始化了分类头)
            }
        """
        features = self.encode_image(image)

        results = {
            "features": features,
            "nutrition": self.regressor(features),
        }

        if self.classifier is not None:
            results["logits"] = self.classifier(features)

        return results

    def zero_shot_classify(
        self,
        image: torch.Tensor,
        class_names: List[str],
        template: str = "a photo of {}",
    ) -> torch.Tensor:
        """零样本分类

        使用 CLIP 的视觉-文本对齐进行零样本食物分类。

        Args:
            image: 输入图像 [B, 3, H, W]
            class_names: 类别名称列表
            template: 文本模板

        Returns:
            分类概率 [B, num_classes]
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP 模型未加载")

        # 构建文本描述
        text_descriptions = [template.format(name) for name in class_names]
        tokens = self.tokenize(text_descriptions).to(image.device)

        # 计算图像和文本特征
        with torch.no_grad():
            image_features = self.encode_image(image)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            logits = (image_features @ text_features.T) * 100.0  # CLIP 的 temperature scaling
            probs = logits.softmax(dim=-1)

        return probs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    try:
        model = CalorieCLIPWrapper(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            freeze_encoder=True,
        ).to(device)

        # 初始化分类头（与主多任务网络一致: 11 类）
        model.init_classifier(num_classes=11)

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 测试前向传播
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            outputs = model(x)
        print(f"营养输出: {outputs['nutrition'].shape}")
        print(f"分类输出: {outputs['logits'].shape}")

    except Exception as e:
        print(f"CalorieCLIP 测试失败（可能未安装 open_clip）: {e}")
        print("请运行: pip install open-clip-torch")
