"""ResNet18 for CIFAR-10 classification (BatchNorm-free).

For SARAH-style algorithms each per-sample gradient ∇f_i(w) must be a
*deterministic* function of ``w``.  BatchNorm, in train mode, mixes
samples through batch statistics and updates running buffers — both of
which break the ``∇f_i`` contract.  Replacing BN with GroupNorm fixes
this: GN is purely sample-wise and has no running buffers.

Architecture changes from torchvision's ResNet-18:
    * 3×3 / stride 1 / padding 1 stem (CIFAR-10 size).
    * No maxpool after the stem (32×32 → 32×32 → 16×16 …).
    * BatchNorm2d → GroupNorm (32 groups when possible).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torchvision.models


def _group_norm_factory(num_groups: int = 32) -> Callable[[int], nn.GroupNorm]:
    def make(num_channels: int) -> nn.GroupNorm:
        # ``num_groups`` must divide ``num_channels``; fall back gracefully.
        groups = num_groups
        while num_channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_channels)

    return make


class ResNet18_32x32(nn.Module):
    """ResNet-18 adapted for 32×32 inputs and BN-free gradients."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        norm_layer = _group_norm_factory()
        base_model = torchvision.models.resnet18(
            weights=None, num_classes=num_classes, norm_layer=norm_layer,
        )

        # Stem: small 3×3 conv, no stride, no maxpool.
        self.embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            base_model.bn1,
            base_model.relu,
        )

        self.body = nn.Sequential(
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
            nn.Flatten(),
        )

        self.head = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.body(x)
        x = self.head(x)
        return x
