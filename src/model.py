import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class LeafNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Identity()
        self.backbone = m
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
