import torch
import torch.nn as nn
from torchvision import models


class BaselineResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["models"]["baseline"]
        
        backbone_name = model_cfg["backbone"]
        pretrained = model_cfg["pretrained"]
        num_classes = model_cfg["num_classes"]
        
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
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
        
        hidden_dim = 512
        dropout = 0.3
        
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
        
        total_loss = cls_loss + 0.5 * cal_loss + 0.3 * weight_loss
        
        return {
            "total": total_loss,
            "cls": cls_loss,
            "calories": cal_loss,
            "weight": weight_loss,
        }


class CalorieCLIPBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["models"]["baseline"]
        num_classes = model_cfg["num_classes"]
        
        try:
            import clip
            self.clip_available = True
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
            feature_dim = 512
        except ImportError:
            self.clip_available = False
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
            feature_dim = 2048
        
        hidden_dim = 512
        dropout = 0.3
        
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
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, x):
        if self.clip_available:
            import clip
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            features = self.clip_model.encode_image(x)
            features = features.float()
        else:
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
        
        total_loss = cls_loss + 0.5 * cal_loss + 0.3 * weight_loss
        
        return {
            "total": total_loss,
            "cls": cls_loss,
            "calories": cal_loss,
            "weight": weight_loss,
        }
