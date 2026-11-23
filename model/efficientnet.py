import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from base import BaseModel


class EfficientNetB0(BaseModel):
    def __init__(self, num_classes=3, use_pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        self.model = efficientnet_b0(weights=weights)
        input_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.model(x)
