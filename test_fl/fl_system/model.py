from __future__ import annotations

import torch
from torch import nn

from models.resnet_cifar import ResNet18


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(dataset: str, num_classes: int, in_channels: int) -> nn.Module:
    """Build model with CIFAR10 configuration aligned to the main FEDCSAP system."""
    if dataset.lower() == "cifar10":
        # 与主系统保持一致：CIFAR10 使用 ResNet18
        return ResNet18(name="Target")
    return SmallCNN(num_classes=num_classes, in_channels=in_channels)
