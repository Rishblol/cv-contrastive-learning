"""Objective functions used by self-supervised training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def nt_xent_loss(
    view_one: torch.Tensor, view_two: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss for paired SimCLR views."""
    if view_one.shape != view_two.shape:
        raise ValueError("SimCLR views must have identical embedding shapes")
    if view_one.shape[0] < 2:
        raise ValueError("NT-Xent requires at least two samples per batch")
    embeddings = F.normalize(torch.cat([view_one, view_two], dim=0), dim=1)
    logits = embeddings @ embeddings.T / temperature
    logits.fill_diagonal_(float("-inf"))
    batch_size = view_one.shape[0]
    positives = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(embeddings.device)
    return F.cross_entropy(logits, positives)
