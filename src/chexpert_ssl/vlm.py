"""Optional Hugging Face VLM adapters and deterministic prompt scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import TARGETS


def _transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:  # pragma: no cover - exercised only in VLM environments
        raise RuntimeError("Install VLM dependencies with: pip install -e '.[vlm]'") from error
    return AutoModel, AutoProcessor


class HuggingFaceVLM(nn.Module):
    """Small adapter for CLIP-compatible radiology checkpoints such as CheXzero."""

    def __init__(self, checkpoint: str, revision: str | None = None) -> None:
        super().__init__()
        auto_model, auto_processor = _transformers()
        self.model = auto_model.from_pretrained(checkpoint, revision=revision)
        self.processor = auto_processor.from_pretrained(checkpoint, revision=revision)
        self.feature_dim = int(getattr(self.model.config, "projection_dim", 512))

    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "get_image_features"):
            raise ValueError("Checkpoint must expose get_image_features for this VLM adapter")
        return self.model.get_image_features(pixel_values=pixel_values)

    @torch.inference_mode()
    def text_features(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        batch = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        if not hasattr(self.model, "get_text_features"):
            raise ValueError("Checkpoint must expose get_text_features for zero-shot scoring")
        return F.normalize(self.model.get_text_features(**batch), dim=-1)


class VLMClassifier(nn.Module):
    def __init__(self, encoder: HuggingFaceVLM, num_labels: int, freeze_encoder: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.feature_dim, num_labels)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder.image_features(pixel_values))


def aggregate_prompt_embeddings(
    model: HuggingFaceVLM, prompt_set: dict[str, dict[str, list[str]]], device: torch.device
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Average normalized template embeddings for each positive/negative concept."""
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for label in TARGETS:
        specification = prompt_set[label]
        positive = F.normalize(model.text_features(specification["positive"], device).mean(dim=0), dim=0)
        negative = F.normalize(model.text_features(specification["negative"], device).mean(dim=0), dim=0)
        result[label] = (positive, negative)
    return result


def zero_shot_scores(image_embeddings: torch.Tensor, concepts: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> np.ndarray:
    """Two-class softmax probability for each pathology; labels never enter this function."""
    images = F.normalize(image_embeddings, dim=-1)
    scores = []
    for label in TARGETS:
        positive, negative = concepts[label]
        logits = torch.stack((images @ negative, images @ positive), dim=-1)
        scores.append(torch.softmax(logits, dim=-1)[:, 1])
    return torch.stack(scores, dim=1).cpu().numpy()
