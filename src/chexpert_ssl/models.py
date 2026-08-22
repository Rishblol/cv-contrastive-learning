"""Encoders and downstream heads shared by SSL and supervised experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def build_encoder(name: str = "resnet50", pretrained: bool = False) -> tuple[nn.Module, int]:
    """Return a ResNet feature encoder without its ImageNet classification layer."""
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        features = model.fc.in_features
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported encoder {name!r}; use resnet18 or resnet50")
    model.fc = nn.Identity()
    return model, features


class SimCLRModel(nn.Module):
    def __init__(self, encoder_name: str, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(images))


class MultiLabelClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.classifier = nn.Linear(feature_dim, num_labels)
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def load_simclr_encoder(self, checkpoint: dict[str, object]) -> None:
        """Load encoder weights from a checkpoint written by train_pretrain.py."""
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError("Checkpoint does not contain SimCLR encoder weights")
        self.encoder.load_state_dict(encoder_state, strict=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))
