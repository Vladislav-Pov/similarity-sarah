import torch.nn as nn

from similarity_sarah.models.resnet import ResNet18_32x32
from similarity_sarah.models.simple_cnn import SimpleCNN

__all__ = ["ResNet18_32x32", "SimpleCNN", "build_model"]


def build_model(name: str, num_classes: int) -> nn.Module:
    """Construct a model by name (the rewrite's two in-scope architectures)."""
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18_32x32":
        return ResNet18_32x32(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
