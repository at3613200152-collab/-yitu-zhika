"""P1-B：五目标（卡路里/重量/蛋白/碳水/脂肪）多任务网络。

基于 ResNet50 骨干（ResNetMultiTask，num_regression_targets=5），
用 meal_macros_v1 manifest 的 5 维 target_stats 做反标准化。
与官方 2 目标 MealNet 分离，避免影响已审计模型。
"""
import torch
from torch import nn

from src.models.resnet_multitask import ResNetMultiTask
from src.training.train_meal_official import MealNet

TARGETS = ['calories', 'mass', 'protein', 'carbohydrate', 'fat']


def target_stats_to_arrays(stats):
    """兼容两种 target_stats 结构：
      官方格式 {mean:[...], std:[...]}；macros 格式 {field:{mean,std,...}}。
    """
    if 'mean' in stats and 'std' in stats:
        return list(stats['mean']), list(stats['std'])
    return [stats[f]['mean'] for f in TARGETS], [stats[f]['std'] for f in TARGETS]


class MealMacrosNet(nn.Module):
    def __init__(self, manifest, pretrained=True, input_channels=3):
        super().__init__()
        self.network = ResNetMultiTask(
            num_classes=len(manifest['category_to_idx']),
            input_channels=input_channels,
            pretrained=pretrained,
            num_regression_targets=5,
        )
        center, scale = target_stats_to_arrays(manifest['target_stats'])
        self.register_buffer('target_center', torch.tensor(center, dtype=torch.float32))
        self.register_buffer('target_scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, images):
        output = self.network(images)
        return output['logits'], output['nutrition'].float() * self.target_scale + self.target_center


def macros_losses(logits, prediction, target, classes, scale, mask):
    """P1-B：带 mask 的逐目标归一化 L1。缺失维度 mask=0，不参与损失，不补 0。

    prediction/target/mask 形状 [B,5]；scale 为 [5]。
    每样本 = 该样本有效目标上 |pred-target|/scale 的均值；再对 batch 求均值。
    分类用 CE，权重 0.2（与官方一致）。
    """
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite regression tensors')
    diff = (prediction - target).abs() * mask / scale.view(1, -1)
    reg = (diff.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    cls = nn.functional.cross_entropy(logits.float(), classes)
    total = reg + 0.2 * cls
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite loss')
    return total, reg


# 复用 MealNet 的模型加载辅助（仅供参考；本模块独立于官方 2 目标模型）
_ = MealNet
