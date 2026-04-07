"""Classification task: cross-entropy loss + accuracy metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ClassificationTask:
    """Wraps the loss function and evaluation logic for classification."""

    def __init__(self) -> None:
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Compute loss and accuracy over the entire *loader*."""
        was_training = model.training
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += self.loss_fn(output, y).item() * x.size(0)
            correct += output.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)

        model.train(was_training)
        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
