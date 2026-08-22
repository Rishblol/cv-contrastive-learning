from __future__ import annotations

import numpy as np
import torch

from chexpert_ssl.losses import nt_xent_loss
from chexpert_ssl.metrics import multilabel_metrics


def test_nt_xent_loss_is_finite_and_differentiable() -> None:
    first = torch.randn(4, 8, requires_grad=True)
    second = torch.randn(4, 8, requires_grad=True)
    loss = nt_xent_loss(first, second)
    loss.backward()
    assert torch.isfinite(loss)
    assert first.grad is not None


def test_multilabel_metrics_reports_per_label_and_macro_values() -> None:
    targets = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4, 0.45], [0.9, 0.8, 0.7, 0.6, 0.55]])
    result = multilabel_metrics(targets, probabilities)
    assert result["macro_auroc"] == 1.0
    assert result["macro_auprc"] == 1.0
