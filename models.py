import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ============================================================
# VGG16 BACKBONE WRAPPER
# ============================================================
class VGG16Backbone(nn.Module):
    """VGG16 backbone wrapper for proper feature extraction."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

# ============================================================
# BACKBONE FACTORY
# ============================================================
def get_backbone(name, pretrained=True):
    """Get backbone network with pretrained weights."""
    if name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        out_features = model.classifier[2].in_features
        # Keep LayerNorm2d and Flatten, only remove the Linear layer
        model.classifier = nn.Sequential(
            model.classifier[0],  # LayerNorm2d
            model.classifier[1],  # Flatten
        )
        
    elif name == 'vgg16':
        # VGG16 backbone - using wrapper class
        model = VGG16Backbone(pretrained=pretrained)
        out_features = 512 * 7 * 7  # 25088
        
    elif name == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)
        out_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif name == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        out_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        
    elif name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        out_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif name == 'densenet121':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        out_features = model.classifier.in_features
        model.classifier = nn.Identity()
        
    elif name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        out_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        
    elif name == 'mobilenet_v3':
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        out_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
    
    elif name == 'vit_b_16':
        # Vision Transformer Base with 16x16 patches
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        out_features = model.heads.head.in_features  # 768
        model.heads = nn.Identity()
    
    elif name == 'swin_t':
        # Swin Transformer Tiny
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        out_features = model.head.in_features  # 768
        model.head = nn.Identity()
        
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    return model, out_features

# ============================================================
# CLASSIFIER HEAD DEFINITIONS
# ============================================================

class SimpleHead(nn.Module):
    """Simple linear classifier head."""
    def __init__(self, in_features, num_classes, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

class MLPHead(nn.Module):
    """Multi-layer perceptron head."""
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

class DeepHead(nn.Module):
    """Deep MLP head with 2 hidden layers."""
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

class AttentionHead(nn.Module):
    """Attention-based classifier head."""
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # For 1D features, attention simplifies to weighted FC
        # This implementation seems to assume x is already flattened or global pooled
        # If it's just a 1D vector, typical attention might not apply directly as spatial/temporal dim
        # But following reference implementation:
        return self.fc(x)

class CosFaceHead(nn.Module):
    """CosFace/ArcFace style head with normalized weights."""
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5, s=30.0, m=0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, labels=None):
        x = self.projection(x)
        # Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        # Scale
        output = self.s * cosine
        return output

def get_head(name, in_features, num_classes, dropout=0.5):
    """Get classifier head by name."""
    heads = {
        'simple': SimpleHead,
        'mlp': MLPHead,
        'deep': DeepHead,
        'attention': AttentionHead,
        'cosface': CosFaceHead
    }
    if name not in heads:
        raise ValueError(f"Unknown head: {name}")
    return heads[name](in_features, num_classes, dropout=dropout)

# ============================================================
# COMPLETE FACE CLASSIFIER MODEL
# ============================================================

class FaceClassifier(nn.Module):
    """Face identification model with configurable backbone and head."""
    
    def __init__(self, num_classes, backbone_name='resnet101', head_name='simple', 
                 pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone_name = backbone_name
        self.head_name = head_name
        self.num_classes = num_classes
        
        # Load backbone
        self.backbone, out_features = get_backbone(backbone_name, pretrained)
        
        # Create classifier head
        self.head = get_head(head_name, out_features, num_classes, dropout)
    
    def forward(self, x):
        features = self.backbone(x)
        # Handle cases where feature extractor might return tuples (like Inception)
        if isinstance(features, tuple):
            features = features[0]
        return self.head(features)
    
    def predict(self, x):
        """Get predictions and probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        return preds, probs
