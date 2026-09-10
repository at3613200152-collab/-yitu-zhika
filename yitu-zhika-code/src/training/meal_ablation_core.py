"""Independent ablation switches and dish-balanced multiview supervision."""
import hashlib
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def arm(fusion='concat', head='positive', paired=False, consistency=0., clip_aux=False,
        class_weight=False):
    return dict(fusion=fusion, head=head, paired=paired, consistency=consistency,
                clip_aux=clip_aux, class_weight=class_weight)


ARMS = {
    'rgb_legacy': arm('rgb', 'legacy'),
    'nir_legacy': arm(head='legacy'),
    'rgb_positive': arm('rgb'),
    'nir_positive': arm(),
    'gate': arm('gate'),
    'factorized': arm(head='factorized'),
    'pair_control': arm(paired=True),
    'consistency': arm(paired=True, consistency=.1),
    'clip_aux': arm(clip_aux=True),
    'combined': arm('gate', 'factorized', paired=True, consistency=.1, clip_aux=True),
    # P2-C：类别纠偏——逐类权重 CE（频率平方根倒数、封顶），与 nir_positive/rgb_positive 同配方只改损失
    'class_weight': arm(class_weight=True),
    'rgb_class_weight': arm('rgb', class_weight=True),
}

# P2-C 预先固定的加权口径（不据测试集调参；如需敏感性分析用 CLI 覆盖并记录配置）
CLASS_WEIGHT_CAP = 4.0
CLASS_WEIGHT_POWER = 0.5


def class_weights_from_manifest(manifest, device=None, cap=CLASS_WEIGHT_CAP, power=CLASS_WEIGHT_POWER):
    """按 **train 划分** 的类别频率计算逐类权重：w_c = min(cap, (N/(K*n_c))**power)。

    - 只在有样本的类别上计算；无样本类别的权重为 0（不产生梯度，也不会被"宣称"支持）；
    - 权重再按"有支持类别的均值=1"归一化，使损失量级与未加权时可比；
    - 返回 (weights_tensor, 诊断信息 dict) 供审计记录。
    """
    n_cls = len(manifest['category_to_idx'])
    counts = [0] * n_cls
    for r in manifest['rows']:
        if r['split'] == 'train':
            counts[r['category_idx']] += 1
    total = sum(counts)
    supported = [c for c in counts if c > 0]
    if total == 0 or not supported:
        raise ValueError('No train rows for class weighting')
    k = len(supported)
    weights = [0.0] * n_cls
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = min(cap, (total / (k * c)) ** power)
    t = torch.tensor(weights, dtype=torch.float32, device=device)
    mean_supported = t[t > 0].mean()
    if float(mean_supported) <= 0:
        raise ValueError('Degenerate class weights')
    t = t / mean_supported
    info = {'cap': float(cap), 'power': float(power), 'train_counts': counts,
            'weights': [round(float(x), 4) for x in t]}
    return t, info


def select_views(row, split, epoch, seed, paired=False):
    views = row.get('views', [dict(path=row['image'], file_sha256=row['image_sha256'])])
    value = int(hashlib.sha256(f'{seed}:{epoch}:{row["dish_id"]}:view'.encode()).hexdigest()[:8], 16)
    first = value % len(views) if split == 'train' else 0
    second = views[(first+1) % len(views)] if split == 'train' and paired and len(views) > 1 else None
    return views[first], second


class ExpandedDataset(Dataset):
    def __init__(self, manifest, split, epoch=0, seed=42, paired=False, semantic_cache=None):
        self.rows = [r for r in manifest['rows'] if r['split'] == split]
        self.split, self.epoch, self.seed, self.paired = split, epoch, seed, paired
        self.cache = semantic_cache
        self.classes = len(manifest['category_to_idx'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        first, second = select_views(row, self.split, self.epoch, self.seed, self.paired)
        flip = (self.split == 'train' and int(hashlib.sha256(
            f'{self.seed}:{self.epoch}:{row["dish_id"]}'.encode()).hexdigest()[:8], 16) < 2**31)

        def image(view):
            with Image.open(ROOT/view['path']) as source:
                im = source.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            if flip:
                im = ImageOps.mirror(im)
            return TF.normalize(TF.to_tensor(im), MEAN, STD)

        def scores(view):
            if self.cache is None:
                return torch.zeros(self.classes)
            return self.cache[view['file_sha256']]

        a = image(first)
        return dict(rgb=a, second_rgb=image(second) if second else a,
                    clip=scores(first), second_clip=scores(second) if second else scores(first),
                    paired=second is not None, targets=torch.tensor(row['targets'], dtype=torch.float32),
                    category=row['category_idx'], dish_id=row['dish_id'])


def ablation_loss(output, target, classes, scale, density_mean, *, second=None,
                  pair_mask=None, consistency_weight=0., class_weight=None):
    def supervised(value, truth, labels):
        reg = ((value['nutrition'].float()-truth).abs()/scale).mean(1)
        # P2-C：class_weight=None 时与历史各臂完全一致；传入时用逐类权重 CE
        cls = F.cross_entropy(value['logits'].float(), labels, weight=class_weight, reduction='none')
        density = torch.zeros_like(reg)
        if value['density'] is not None:
            valid = truth[:, 1] > 0
            density[valid] = (value['density'][valid]-truth[valid, 0]/truth[valid, 1]).abs()/density_mean
        return reg+.2*cls+.1*density, reg

    per_dish, reg = supervised(output, target, classes)
    consistency = per_dish.new_zeros(())
    if second is not None:
        if pair_mask is None or int(pair_mask.sum()) != len(second['nutrition']):
            raise ValueError('Second predictions must align with same-dish pair mask')
        extra, extra_reg = supervised(second, target[pair_mask], classes[pair_mask])
        # Each dish has equal weight; two video frames do not count as two meals.
        per_dish = per_dish.clone()
        reg = reg.clone()
        per_dish[pair_mask] = (per_dish[pair_mask]+extra)/2
        reg[pair_mask] = (reg[pair_mask]+extra_reg)/2
        consistency = ((output['nutrition'][pair_mask]-second['nutrition']).abs()/scale).mean()
    total = per_dish.mean()+consistency_weight*consistency
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite ablation loss')
    return total, dict(reg_normalized_l1=reg.mean(), consistency=consistency)
