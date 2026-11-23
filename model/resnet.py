import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from base import BaseModel


class ResNet152(BaseModel):

    def __init__(self, num_classes=3, use_pretrained=True):
        super(BaseModel, self).__init__()
        weights = ResNet152_Weights.DEFAULT if use_pretrained else None
        self.model = resnet152(weights=weights)

        # Replace the ImageNet classifier with the dataset classifier.
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)


class ResNet101(BaseModel):

    def __init__(self, num_classes=3, use_pretrained=True):
        super(BaseModel, self).__init__()
        weights = ResNet101_Weights.DEFAULT if use_pretrained else None
        self.model = resnet101(weights=weights)

        # Replace the ImageNet classifier with the dataset classifier.
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)


class ResNet50(BaseModel):

    def __init__(self, num_classes=3, use_pretrained=True):
        super(BaseModel, self).__init__()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        self.model = resnet50(weights=weights)

        # Replace the ImageNet classifier with the dataset classifier.
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)


class ResNet34(BaseModel):

    def __init__(self, num_classes=3, use_pretrained=True):
        super(BaseModel, self).__init__()
        weights = ResNet34_Weights.DEFAULT if use_pretrained else None
        self.model = resnet34(weights=weights)

        # Replace the ImageNet classifier with the dataset classifier.
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)


class ResNet18(BaseModel):

    def __init__(self, num_classes=3, use_pretrained=True):
        super(BaseModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.model = resnet18(weights=weights)

        # Replace the ImageNet classifier with the dataset classifier.
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)
