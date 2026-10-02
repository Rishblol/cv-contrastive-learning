"""Encoders and downstream heads shared by SSL and supervised experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as vision_models


def build_encoder(name: str = "resnet50", pretrained: bool = False) -> tuple[nn.Module, int]:
    """Return a supported torchvision backbone that emits one feature vector per image."""
    constructors = {
        "resnet18": (vision_models.resnet18, vision_models.ResNet18_Weights),
        "resnet50": (vision_models.resnet50, vision_models.ResNet50_Weights),
        "densenet121": (vision_models.densenet121, vision_models.DenseNet121_Weights),
        "efficientnet_b0": (vision_models.efficientnet_b0, vision_models.EfficientNet_B0_Weights),
        "vit_b_16": (vision_models.vit_b_16, vision_models.ViT_B_16_Weights),
    }
    if name not in constructors:
        supported = ", ".join(constructors)
        raise ValueError(f"Unsupported encoder {name!r}; choose one of: {supported}")

    constructor, weight_enum = constructors[name]
    model = constructor(weights=weight_enum.DEFAULT if pretrained else None)
    if name.startswith("resnet"):
        features = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "densenet121":
        features = model.classifier.in_features
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    else:
        features = model.heads.head.in_features
        model.heads = nn.Identity()
    return model, features


class SimCLRModel(nn.Module):
    def __init__(self, encoder_name: str, projection_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name, pretrained=pretrained)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(images))


class MultiLabelClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        freeze_encoder: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name, pretrained=pretrained)
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
