"""Five image-only self-supervised objectives sharing the project encoder."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from chexpert_ssl.models import build_encoder


METHODS = ("simclr", "moco", "byol", "nnclr", "swav")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def _normalize(output: torch.Tensor) -> torch.Tensor:
    return F.normalize(output.float(), dim=1)


def _nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = first.shape[0]
    z = torch.cat((first, second))
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat((torch.arange(batch, 2 * batch), torch.arange(batch))).to(z.device)
    return F.cross_entropy(logits, targets)


def _info_nce(prediction: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.cross_entropy(prediction @ positive.T / temperature,
                           torch.arange(prediction.shape[0], device=prediction.device))


@torch.no_grad()
def _ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_parameter.lerp_(online_parameter.detach(), 1.0 - momentum)


class SSLMethod(nn.Module):
    def after_optimizer_step(self, progress: float) -> None:
        del progress


class SimCLR(SSLMethod):
    def __init__(self, encoder_name: str, projection_dim: int, hidden_dim: int, temperature: float):
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.projector = _mlp(feature_dim, hidden_dim, projection_dim)
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch = first.shape[0]
        projections = _normalize(self.projector(self.encoder(torch.cat((first, second)))))
        return _nt_xent(projections[:batch], projections[batch:], self.temperature)


class MoCo(SSLMethod):
    def __init__(self, encoder_name: str, projection_dim: int, hidden_dim: int,
                 temperature: float, queue_size: int, momentum: float):
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.projector = _mlp(feature_dim, hidden_dim, projection_dim)
        self.key_encoder = copy.deepcopy(self.encoder)
        self.key_projector = copy.deepcopy(self.projector)
        for parameter in (*self.key_encoder.parameters(), *self.key_projector.parameters()):
            parameter.requires_grad_(False)
        self.temperature, self.momentum, self.queue_size = temperature, momentum, queue_size
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, projection_dim), dim=1))
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _keys(self, images: torch.Tensor) -> torch.Tensor:
        permutation = torch.randperm(images.shape[0], device=images.device)
        keys = _normalize(self.key_projector(self.key_encoder(images[permutation])))
        return keys[torch.argsort(permutation)]

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        count, pointer = keys.shape[0], int(self.queue_pointer)
        positions = (pointer + torch.arange(count, device=keys.device)) % self.queue_size
        self.queue[positions] = keys
        self.queue_pointer[0] = (pointer + count) % self.queue_size

    def _loss(self, query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        positive = (query * key).sum(dim=1, keepdim=True)
        logits = torch.cat((positive, query @ queue.T), dim=1) / self.temperature
        return F.cross_entropy(logits, torch.zeros(query.shape[0], dtype=torch.long, device=query.device))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        queue = self.queue.detach().clone()
        q1 = _normalize(self.projector(self.encoder(first)))
        q2 = _normalize(self.projector(self.encoder(second)))
        k1, k2 = self._keys(first), self._keys(second)
        loss = 0.5 * (self._loss(q1, k2, queue) + self._loss(q2, k1, queue))
        self._enqueue(torch.cat((k1, k2)))
        return loss

    def after_optimizer_step(self, progress: float) -> None:
        del progress
        _ema_update(self.key_encoder, self.encoder, self.momentum)
        _ema_update(self.key_projector, self.projector, self.momentum)


class BYOL(SSLMethod):
    def __init__(self, encoder_name: str, projection_dim: int, hidden_dim: int, momentum: float):
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.projector = _mlp(feature_dim, hidden_dim, projection_dim)
        self.predictor = _mlp(projection_dim, hidden_dim, projection_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for parameter in (*self.target_encoder.parameters(), *self.target_projector.parameters()):
            parameter.requires_grad_(False)
        self.base_momentum = momentum

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        p1 = self.predictor(self.projector(self.encoder(first)))
        p2 = self.predictor(self.projector(self.encoder(second)))
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(first))
            t2 = self.target_projector(self.target_encoder(second))
        loss1 = 2 - 2 * (_normalize(p1) * _normalize(t2)).sum(dim=1).mean()
        loss2 = 2 - 2 * (_normalize(p2) * _normalize(t1)).sum(dim=1).mean()
        return loss1 + loss2

    def after_optimizer_step(self, progress: float) -> None:
        momentum = 1 - (1 - self.base_momentum) * (math.cos(math.pi * progress) + 1) / 2
        _ema_update(self.target_encoder, self.encoder, momentum)
        _ema_update(self.target_projector, self.projector, momentum)


class NNCLR(SSLMethod):
    def __init__(self, encoder_name: str, projection_dim: int, hidden_dim: int,
                 temperature: float, queue_size: int):
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.predictor = _mlp(projection_dim, hidden_dim, projection_dim)
        self.temperature, self.queue_size = temperature, queue_size
        self.register_buffer("support", F.normalize(torch.randn(queue_size, projection_dim), dim=1))
        self.register_buffer("support_filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("support_pointer", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _nearest(self, embeddings: torch.Tensor) -> torch.Tensor:
        if int(self.support_filled) < self.queue_size:
            return embeddings
        return self.support[(embeddings @ self.support.T).argmax(dim=1)]

    @torch.no_grad()
    def _enqueue(self, embeddings: torch.Tensor) -> None:
        count, pointer = embeddings.shape[0], int(self.support_pointer)
        positions = (pointer + torch.arange(count, device=embeddings.device)) % self.queue_size
        self.support[positions] = embeddings
        self.support_pointer[0] = (pointer + count) % self.queue_size
        self.support_filled[0] = min(self.queue_size, int(self.support_filled) + count)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        h1, h2 = self.projector(self.encoder(first)), self.projector(self.encoder(second))
        z1, z2 = _normalize(h1.detach()), _normalize(h2.detach())
        p1, p2 = _normalize(self.predictor(h1)), _normalize(self.predictor(h2))
        loss = 0.5 * (_info_nce(p2, self._nearest(z1), self.temperature)
                      + _info_nce(p1, self._nearest(z2), self.temperature))
        self._enqueue(torch.cat((z1, z2)))
        return loss


@torch.no_grad()
def _sinkhorn(scores: torch.Tensor, epsilon: float, iterations: int = 3) -> torch.Tensor:
    assignments = torch.exp(scores / epsilon).T
    assignments /= assignments.sum().clamp_min(1e-12)
    prototypes, batch = assignments.shape
    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
        assignments /= prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
        assignments /= batch
    return (assignments * batch).T


class SwAV(SSLMethod):
    def __init__(self, encoder_name: str, projection_dim: int, hidden_dim: int,
                 temperature: float, prototype_count: int, epsilon: float, freeze_steps: int):
        super().__init__()
        self.encoder, feature_dim = build_encoder(encoder_name)
        self.projector = _mlp(feature_dim, hidden_dim, projection_dim)
        self.prototypes = nn.Parameter(F.normalize(torch.randn(prototype_count, projection_dim), dim=1))
        self.temperature, self.epsilon, self.freeze_prototype_steps = temperature, epsilon, freeze_steps
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, dim=1))
        weight = self.prototypes.detach() if int(self.step) < self.freeze_prototype_steps else self.prototypes
        batch = first.shape[0]
        embeddings = _normalize(self.projector(self.encoder(torch.cat((first, second)))))
        scores = embeddings @ weight.float().T
        first_scores, second_scores = scores[:batch], scores[batch:]
        first_assign = _sinkhorn(first_scores.detach(), self.epsilon)
        second_assign = _sinkhorn(second_scores.detach(), self.epsilon)
        loss1 = -(first_assign * F.log_softmax(second_scores / self.temperature, dim=1)).sum(1).mean()
        loss2 = -(second_assign * F.log_softmax(first_scores / self.temperature, dim=1)).sum(1).mean()
        return 0.5 * (loss1 + loss2)

    def after_optimizer_step(self, progress: float) -> None:
        del progress
        self.step += 1


def build_ssl_method(name: str, encoder_name: str, config: dict) -> SSLMethod:
    """Build one method from the common YAML SSL settings."""
    if name not in METHODS:
        raise ValueError(f"Unknown SSL method {name!r}; choose from {', '.join(METHODS)}")
    common = (encoder_name, int(config.get("projection_dim", 128)), int(config.get("hidden_dim", 1024)))
    temperature = float(config.get("temperatures", {}).get(name, config.get("temperature", 0.2)))
    if name == "simclr":
        return SimCLR(*common, temperature)
    if name == "moco":
        return MoCo(*common, temperature, int(config.get("queue_size", 4096)),
                    float(config.get("momentum", 0.99)))
    if name == "byol":
        return BYOL(*common, float(config.get("momentum", 0.99)))
    if name == "nnclr":
        return NNCLR(*common, temperature, int(config.get("queue_size", 8192)))
    return SwAV(*common, temperature, int(config.get("prototype_count", 300)),
                float(config.get("sinkhorn_epsilon", 0.05)), int(config.get("prototype_freeze_steps", 100)))


def encoder_state_dict(method: SSLMethod) -> dict[str, torch.Tensor]:
    """Return the online encoder weights saved for downstream transfer."""
    return method.encoder.state_dict()
