"""
消融实验脚本
==============
系统化地评估各优化模块的贡献。

消融实验设计:
    1. Baseline: Pix2Pix(U-Net) + ResNet50多任务
    2. + Advanced Generator (Pix2PixHD/扩散)
    3. + Multi-band NIR (5波段)
    4. + Spatial Attention (CBAM)
    5. + Cooking Method Head
    6. + Food Segmentation Preprocessing
    7. Full Model: 所有优化叠加

每项实验记录:
    - 图像指标: PSNR, SSIM (阶段一)
    - 营养指标: MAPE, RMSE, MAE (阶段二)
    - 分类精度: Top-1, Top-5 (阶段二)
    - 推理速度: FPS, GPU显存占用
"""

import os
import yaml
import json
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """消融实验配置

    Attributes:
        name: 实验名称
        use_advanced_generator: 是否使用高级生成器
        use_multi_band: 是否使用多波段NIR
        use_attention: 是否使用空间注意力
        use_cooking_head: 是否使用烹饪方式头
        use_segmentation: 是否使用食物分割
        generator_type: 高级生成器类型 ("pix2pix" / "pix2pixhd" / "diffusion")
        num_bands: 多波段数量
    """
    name: str = "baseline"
    use_advanced_generator: bool = False
    use_multi_band: bool = False
    use_attention: bool = False
    use_cooking_head: bool = False
    use_segmentation: bool = False
    generator_type: str = "pix2pix"
    num_bands: int = 1


# 预定义的消融实验组
ABLATION_GROUPS = [
    AblationConfig(name="1_baseline"),
    AblationConfig(name="2_advanced_gen", use_advanced_generator=True, generator_type="pix2pixhd"),
    AblationConfig(name="3_multi_band", use_multi_band=True, num_bands=5),
    AblationConfig(name="4_attention", use_attention=True),
    AblationConfig(name="5_cooking_head", use_cooking_head=True),
    AblationConfig(name="6_segmentation", use_segmentation=True),
    AblationConfig(name="7_full",
                   use_advanced_generator=True, generator_type="pix2pixhd",
                   use_multi_band=True, num_bands=5,
                   use_attention=True, use_cooking_head=True,
                   use_segmentation=True),
]


@dataclass
class ExperimentResult:
    """实验结果

    Attributes:
        config_name: 实验配置名称
        image_metrics: 图像质量指标 {psnr, ssim}
        nutrition_metrics: 营养指标 {cal_mape, cal_rmse, cal_mae, weight_mape, ...}
        classification_metrics: 分类指标 {top1_acc, top5_acc}
        inference_fps: 推理速度 (FPS)
        gpu_memory_mb: GPU显存占用 (MB)
        training_time_s: 训练时间 (秒)
    """
    config_name: str = ""
    image_metrics: Dict[str, float] = field(default_factory=dict)
    nutrition_metrics: Dict[str, float] = field(default_factory=dict)
    classification_metrics: Dict[str, float] = field(default_factory=dict)
    inference_fps: float = 0.0
    gpu_memory_mb: float = 0.0
    training_time_s: float = 0.0


class AblationRunner:
    """消融实验执行器

    按配置顺序执行消融实验，记录结果并生成对比报告。

    Args:
        config_path: 基础配置文件路径
        output_dir: 结果输出目录
        device: 计算设备
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        output_dir: str = "./output/ablation",
        device: str = "cuda",
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[ExperimentResult] = []

        # 加载基础配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)

        os.makedirs(output_dir, exist_ok=True)

    def build_model(self, config: AblationConfig) -> torch.nn.Module:
        """根据消融配置构建模型

        Args:
            config: 消融实验配置

        Returns:
            构建好的模型
        """
        # 延迟导入避免循环依赖
        from models.generator.pix2pix import Pix2PixModel
        from models.multitask.resnet_multitask import ResNetMultiTask
        from optimizations.advanced_generator import create_advanced_generator

        # 阶段一模型（生成器）
        if config.use_advanced_generator:
            generator = create_advanced_generator(
                method=config.generator_type,
                input_channels=3,
                output_channels=config.num_bands if config.use_multi_band else 1,
            )
        else:
            generator = Pix2PixModel(
                input_channels=3,
                output_channels=config.num_bands if config.use_multi_band else 1,
            )

        # 阶段二模型（多任务网络）
        input_ch = 3 + config.num_bands if config.use_multi_band else 4
        multitask = ResNetMultiTask(
            num_classes=self.base_config.get('multitask', {}).get('num_classes', 61),
            input_channels=input_ch,
            pretrained=True,
        )

        return {
            "generator": generator,
            "multitask": multitask,
            "config": config,
        }

    def measure_inference_speed(
        self,
        model: torch.nn.Module,
        input_size: tuple = (1, 3, 256, 256),
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> float:
        """测量推理速度

        Args:
            model: 模型
            input_size: 输入尺寸
            num_warmup: 预热次数
            num_iterations: 测量次数

        Returns:
            FPS
        """
        model.eval()
        x = torch.randn(*input_size).to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)

        # 测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        fps = num_iterations / elapsed
        return fps

    def measure_gpu_memory(self, model: torch.nn.Module, input_size: tuple = (1, 3, 256, 256)) -> float:
        """测量GPU显存占用

        Returns:
            显存占用 (MB)
        """
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(*input_size).to(self.device)
        model.eval()
        with torch.no_grad():
            _ = model(x)
        return torch.cuda.max_memory_allocated() / 1024 / 1024

    def run_single(self, config: AblationConfig) -> ExperimentResult:
        """执行单个消融实验

        注意: 此方法为框架代码，实际训练和评估需要配合真实数据。
        此处仅演示流程，实际指标需要训练后填入。

        Args:
            config: 消融实验配置

        Returns:
            实验结果
        """
        result = ExperimentResult(config_name=config.name)

        print(f"\n{'='*60}")
        print(f"消融实验: {config.name}")
        print(f"{'='*60}")
        print(f"  高级生成器: {config.use_advanced_generator} ({config.generator_type})")
        print(f"  多波段NIR:  {config.use_multi_band} ({config.num_bands}波段)")
        print(f"  空间注意力: {config.use_attention}")
        print(f"  烹饪方式头: {config.use_cooking_head}")
        print(f"  食物分割:   {config.use_segmentation}")

        # 构建模型
        models = self.build_model(config)

        # 测量推理速度
        try:
            fps = self.measure_inference_speed(models["multitask"])
            result.inference_fps = fps
            print(f"  推理速度: {fps:.1f} FPS")
        except Exception as e:
            print(f"  推理速度测量失败: {e}")

        # 测量GPU显存
        try:
            mem = self.measure_gpu_memory(models["multitask"])
            result.gpu_memory_mb = mem
            print(f"  GPU显存: {mem:.0f} MB")
        except Exception as e:
            print(f"  显存测量失败: {e}")

        # 实际训练和评估需要在此处执行
        # 以下为占位逻辑，需要替换为真实评估
        print("  [提示] 实际指标需训练后填入")

        return result

    def run_all(self, configs: Optional[List[AblationConfig]] = None):
        """执行所有消融实验

        Args:
            configs: 消融配置列表 (默认使用预定义组)
        """
        if configs is None:
            configs = ABLATION_GROUPS

        for config in configs:
            result = self.run_single(config)
            self.results.append(result)

        # 保存结果
        self.save_results()

        # 生成对比报告
        report = self.generate_report()
        print(report)

    def save_results(self, filename: str = "ablation_results.json"):
        """保存实验结果到JSON"""
        results_dict = []
        for r in self.results:
            results_dict.append({
                "config_name": r.config_name,
                "image_metrics": r.image_metrics,
                "nutrition_metrics": r.nutrition_metrics,
                "classification_metrics": r.classification_metrics,
                "inference_fps": r.inference_fps,
                "gpu_memory_mb": r.gpu_memory_mb,
                "training_time_s": r.training_time_s,
            })

        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {save_path}")

    def generate_report(self) -> str:
        """生成消融实验对比报告"""
        lines = ["\n" + "=" * 80, "消融实验对比报告", "=" * 80]

        # 表头
        header = f"{'实验':<20} {'PSNR':>8} {'SSIM':>8} {'Cal MAPE':>10} {'Top-1':>8} {'FPS':>8} {'GPU MB':>8}"
        lines.append(header)
        lines.append("-" * 80)

        for r in self.results:
            psnr = r.image_metrics.get('psnr', 0)
            ssim = r.image_metrics.get('ssim', 0)
            cal_mape = r.nutrition_metrics.get('cal_mape', 0)
            top1 = r.classification_metrics.get('top1_acc', 0)
            fps = r.inference_fps
            mem = r.gpu_memory_mb

            line = f"{r.config_name:<20} {psnr:>8.2f} {ssim:>8.4f} {cal_mape:>10.2f}% {top1:>8.2f}% {fps:>8.1f} {mem:>8.0f}"
            lines.append(line)

        lines.append("=" * 80)
        return "\n".join(lines)


if __name__ == "__main__":
    runner = AblationRunner(
        config_path="configs/default.yaml",
        output_dir="./output/ablation",
    )
    runner.run_all()
