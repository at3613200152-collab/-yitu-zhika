"""Physical heads for independent ablations; no claims of calibrated uncertainty."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.resnet_multitask import ResNetMultiTask


class NutritionHead(nn.Module):
    def __init__(self, mode, center, scale, density_mean):
        super().__init__()
        if mode not in ('legacy', 'positive', 'factorized'):
            raise ValueError('Unknown nutrition head')
        if min(center) <= 0 or min(scale) <= 0 or density_mean <= 0:
            raise ValueError('Positive training statistics required')
        self.mode = mode
        self.register_buffer('center', torch.tensor(center, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        self.register_buffer('density_mean', torch.tensor(density_mean, dtype=torch.float32))

    def forward(self, raw):
        raw = raw.float()
        if self.mode == 'legacy':
            return raw*self.scale+self.center, None
        positive = F.softplus(raw+math.log(math.expm1(1.)))
        if self.mode == 'positive':
            return positive*self.center, None
        density = positive[:, 0]*self.density_mean
        mass = positive[:, 1]*self.center[1]
        return torch.stack((mass*density, mass), dim=1), density


class MealAblationNet(nn.Module):
    """Optional sample-wise NIR gate; accepts externally frozen NIR predictions."""
    def __init__(self, manifest, fusion='rgb', head='legacy', pretrained=True, clip_aux=False):
        super().__init__()
        if fusion not in ('rgb', 'concat', 'gate'):
            raise ValueError('Unknown fusion')
        self.fusion, self.clip_aux = fusion, clip_aux
        classes = len(manifest['category_to_idx'])
        self.network = ResNetMultiTask(num_classes=classes, input_channels=3,
                                      pretrained=pretrained, num_regression_targets=2)
        if fusion != 'rgb':
            original = self.network.features[0]
            expanded = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :3].copy_(original.weight)
            self.network.features[0] = expanded
        self.physical = NutritionHead(head, manifest['target_stats']['mean'],
                                      manifest['target_stats']['std'], manifest['density_stats']['mean'])
        self.gate_net = nn.Linear(8, 1)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
        self.semantic_adapter = nn.Linear(classes, self.network.feature_dim, bias=False)
        nn.init.zeros_(self.semantic_adapter.weight)
        self.gate_net.requires_grad_(fusion == 'gate')
        self.semantic_adapter.requires_grad_(clip_aux)

    def forward(self, rgb, nir=None, clip_scores=None, gate_override=None):
        gate = torch.zeros(len(rgb), 1, device=rgb.device)
        images = rgb
        if self.fusion != 'rgb':
            if nir is None or nir.shape != (len(rgb), 1, *rgb.shape[-2:]):
                raise ValueError('Matching predicted NIR required')
            if not torch.isfinite(nir).all():
                raise FloatingPointError('Nonfinite predicted NIR')
            gate = torch.ones_like(gate)
            if self.fusion == 'gate':
                context = torch.cat((rgb.float().mean((2, 3)), rgb.float().std((2, 3)),
                                     nir.float().mean((2, 3)), nir.float().std((2, 3))), dim=1)
                gate = torch.sigmoid(self.gate_net(context)).float()
            if gate_override is not None:
                if not 0 <= gate_override <= 1:
                    raise ValueError('Gate override must be between zero and one')
                gate = torch.full_like(gate, float(gate_override))
            images = torch.cat((rgb, nir*gate[:, :, None, None]), dim=1)
        features = self.network.avgpool(self.network.features(images)).flatten(1)
        if self.clip_aux:
            if clip_scores is None or clip_scores.shape != (len(rgb), self.semantic_adapter.in_features):
                raise ValueError('Frozen CLIP semantic scores required')
            if not torch.isfinite(clip_scores).all():
                raise FloatingPointError('Nonfinite CLIP scores')
            features = features+self.semantic_adapter(clip_scores.float())
        logits = self.network.classifier(features)
        nutrition, density = self.physical(self.network.regressor(features))
        return dict(logits=logits, nutrition=nutrition, density=density, gate=gate)
