import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["models"]["multitask"]
        
        backbone_name = model_cfg["backbone"]
        pretrained = model_cfg["pretrained"]
        in_channels = model_cfg["in_channels"]
        num_classes = model_cfg["num_classes"]
        hidden_dim = model_cfg["hidden_dim"]
        dropout = model_cfg["dropout"]
        
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone_name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet101(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        if in_channels != 3:
            old_conv = base_model.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if in_channels > 3:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    for i in range(3, in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, 0]
            base_model.conv1 = new_conv
        
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.calorie_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.weight_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.lambda_cls = model_cfg["lambda_cls"]
        self.lambda_cal = model_cfg["lambda_cal"]
        self.lambda_weight = model_cfg["lambda_weight"]
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        cls_logits = self.classifier(features)
        calorie_pred = self.calorie_head(features).squeeze(-1)
        weight_pred = self.weight_head(features).squeeze(-1)
        
        return {
            "cls_logits": cls_logits,
            "calories": calorie_pred,
            "weight": weight_pred,
        }
    
    def compute_loss(self, predictions, labels):
        cls_loss = self.ce_loss(predictions["cls_logits"], labels["class"])
        cal_loss = self.l1_loss(predictions["calories"], labels["calories"])
        weight_loss = self.l1_loss(predictions["weight"], labels["weight"])
        
        total_loss = (
            self.lambda_cls * cls_loss
            + self.lambda_cal * cal_loss
            + self.lambda_weight * weight_loss
        )
        
        return {
            "total": total_loss,
            "cls": cls_loss,
            "calories": cal_loss,
            "weight": weight_loss,
        }
