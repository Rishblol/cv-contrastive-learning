from __future__ import annotations

import gc

import pytest
import torch

from chexpert_ssl.models import MultiLabelClassifier, build_encoder


@pytest.mark.parametrize(
    ("name", "feature_dim"),
    [
        ("resnet18", 512),
        ("resnet50", 2048),
        ("densenet121", 1024),
        ("efficientnet_b0", 1280),
        ("vit_b_16", 768),
    ],
)
def test_backbone_feature_interface(name: str, feature_dim: int) -> None:
    encoder, actual_dim = build_encoder(name)
    encoder.eval()
    classifier = torch.nn.Linear(feature_dim, 5)
    with torch.inference_mode():
        features = encoder(torch.zeros(1, 3, 224, 224))
        logits = classifier(features)
    assert actual_dim == feature_dim
    assert features.shape == (1, feature_dim)
    assert logits.shape == (1, 5)
    del encoder, classifier, features
    gc.collect()


def test_multilabel_classifier_uses_the_shared_feature_interface() -> None:
    model = MultiLabelClassifier("resnet18", num_labels=5).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 5)


def test_unknown_backbone_lists_supported_choices() -> None:
    with pytest.raises(ValueError, match="vit_b_16"):
        build_encoder("unknown")
