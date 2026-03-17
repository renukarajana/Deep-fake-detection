from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


ClassifierHeadType = Literal["linear"]


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    classifier_head: ClassifierHeadType = "linear"
    dropout: float = 0.2


def build_model(config: ModelConfig) -> nn.Module:
    if config.backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if config.pretrained else None)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, config.num_classes),
        )
        return net

    raise ValueError(f"Unsupported backbone: {config.backbone}")


