"""Small CNN for CIFAR-10 classification."""

from __future__ import annotations

import torch.nn as nn


class SimpleCNN(nn.Module):
    """A lightweight CNN without batch-norm (avoids stochastic forward pass).

    Architecture: Conv32 → ReLU → Pool → Conv64 → ReLU → Pool → FC256 → ReLU → FC(num_classes)
    Input size: 3 × 32 × 32 (CIFAR-10).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
