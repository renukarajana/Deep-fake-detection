from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class VideoModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    feature_dim: int = 512  # resnet18 penultimate features
    lstm_hidden: int = 256
    lstm_layers: int = 1
    bidirectional: bool = False
    num_classes: int = 2
    dropout: float = 0.2


class FrameFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            modules = list(net.children())[:-1]  # remove fc, keep up to avgpool
            self.encoder = nn.Sequential(*modules)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        feats = self.encoder(x)  # [B, C, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, C]
        return feats


class CnnLstmVideoClassifier(nn.Module):
    def __init__(self, cfg: VideoModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FrameFeatureExtractor(cfg.backbone, cfg.pretrained)
        assert self.feature_extractor.feature_dim == cfg.feature_dim
        self.lstm = nn.LSTM(
            input_size=cfg.feature_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        lstm_out = cfg.lstm_hidden * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(lstm_out, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W]
        b, t = x.size(0), x.size(1)
        x_flat = x.view(b * t, *x.size()[2:])  # [B*T, 3, H, W]
        feats = self.feature_extractor(x_flat)  # [B*T, F]
        feats = feats.view(b, t, -1)  # [B, T, F]
        lstm_out, _ = self.lstm(feats)  # [B, T, H]
        last = lstm_out[:, -1, :]  # use last timestep
        logits = self.head(last)  # [B, C]
        return logits






