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
    def __init__(self, manifest, pretrained=True, input_channels=3, nonneg=False):
        """nonneg=True：五个物理量（热量/重量/蛋白/碳水/脂肪）用 softplus 参数化，输出恒非负（方案 §7.1 B）。

        注意完整映射：prediction_phys = softplus(raw)，raw 为回归头原始输出；
        训练损失仍在"归一化尺度"上计算（(pred-target)/scale），因此与非负化前的损失量级可比。
        初始化时把回归头偏置设为 inv_softplus(center)，使初始预测≈训练集均值，避免从 0 附近起步。
        """
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
        self.nonneg = bool(nonneg)
        if self.nonneg:
            with torch.no_grad():
                center_t = self.target_center.clone().clamp(min=1e-3)
                # 数值稳定的 softplus 逆：大值用 y+log1p(-exp(-y))，小值用 log(expm1(y))
                # （直接 log(expm1(y)) 在 y>88 时 float32 溢出为 inf，导致初始预测 inf）
                inv = torch.where(center_t > 20.0,
                                  center_t + torch.log1p(-torch.exp(-center_t)),
                                  torch.log(torch.expm1(center_t)))
                # 回归头可能是 Sequential，取最后一层带 bias 的 Linear
                last_linear = None
                for module in self.network.regressor.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                if last_linear is None or last_linear.bias is None:
                    raise RuntimeError('无法定位回归头最后一层 Linear(bias)，非负初始化失败')
                if last_linear.bias.numel() != inv.numel():
                    raise RuntimeError(f'回归头输出维度 {last_linear.bias.numel()} 与目标数 {inv.numel()} 不一致')
                last_linear.bias.copy_(inv)

    def forward(self, images):
        output = self.network(images)
        raw = output['nutrition'].float()
        if self.nonneg:
            prediction = nn.functional.softplus(raw)
        else:
            prediction = raw * self.target_scale + self.target_center
        return output['logits'], prediction


def macros_losses(logits, prediction, target, classes, scale, mask, class_weight=None):
    """P1-B：带 mask 的逐目标归一化 L1。缺失维度 mask=0，不参与损失，不补 0。

    prediction/target/mask 形状 [B,5]；scale 为 [5]；class_weight 可选（方案 §7.1 C：逐类权重 CE）。
    每样本 = 该样本有效目标上 |pred-target|/scale 的均值；再对 batch 求均值。
    分类用 CE，权重 0.2（与官方一致）。
    """
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite regression tensors')
    diff = (prediction - target).abs() * mask / scale.view(1, -1)
    reg = (diff.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    cls = nn.functional.cross_entropy(logits.float(), classes, weight=class_weight)
    total = reg + 0.2 * cls
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite loss')
    return total, reg


# 复用 MealNet 的模型加载辅助（仅供参考；本模块独立于官方 2 目标模型）
_ = MealNet
