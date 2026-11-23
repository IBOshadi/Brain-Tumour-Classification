import torch.nn as nn
from torchvision.models import SqueezeNet1_0_Weights, squeezenet1_0
from base import BaseModel


class SqueezeNet(BaseModel):
    def __init__(self, num_classes=3, use_pretrained=False):
        super(BaseModel, self).__init__()
        weights = SqueezeNet1_0_Weights.DEFAULT if use_pretrained else None
        self.model = squeezenet1_0(weights=weights)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model.forward(x)
