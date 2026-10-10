from __future__ import annotations

import pytest
import torch

from chexpert_ssl.ssl_methods import METHODS, _sinkhorn, build_ssl_method


@pytest.mark.parametrize("method_name", METHODS)
def test_ssl_method_forward_backward_and_encoder_export(method_name: str) -> None:
    config = {
        "projection_dim": 16,
        "hidden_dim": 32,
        "temperatures": {method_name: 0.2},
        "queue_size": 32,
        "momentum": 0.99,
        "prototype_count": 16,
        "prototype_freeze_steps": 1,
    }
    model = build_ssl_method(method_name, "resnet18", config).train()
    first = torch.rand(2, 3, 64, 64)
    second = torch.rand(2, 3, 64, 64)
    loss = model(first, second)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    model.after_optimizer_step(0.5)
    assert set(model.encoder.state_dict())


def test_sinkhorn_is_stable_for_half_precision_extreme_scores() -> None:
    scores = torch.tensor([[100, -100, 0], [-100, 100, 0]], dtype=torch.float16)
    assignment = _sinkhorn(scores, epsilon=0.05)
    assert assignment.dtype == torch.float32
    assert torch.isfinite(assignment).all()
    torch.testing.assert_close(assignment.sum(dim=1), torch.ones(2))
