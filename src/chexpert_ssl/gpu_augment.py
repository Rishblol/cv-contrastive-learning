"""Batched X-ray augmentations applied on the training device."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _uniform(batch: int, bounds: tuple[float, float], device: torch.device) -> torch.Tensor:
    return torch.rand(batch, device=device) * (bounds[1] - bounds[0]) + bounds[0]


@torch.no_grad()
def augment_grayscale_batch(
    images: torch.Tensor,
    image_size: int,
    config: dict,
    channels_last: bool = True,
) -> torch.Tensor:
    """Augment uint8 images shaped (B,H,W), returning normalized RGB tensors."""
    if images.ndim != 3 or images.dtype != torch.uint8:
        raise ValueError("Expected a uint8 image batch shaped (batch, height, width)")
    device = images.device
    batch = images.shape[0]
    x = images.to(dtype=torch.float32).unsqueeze(1).div_(255.0)

    area = _uniform(batch, tuple(config.get("scale", (0.6, 1.0))), device)
    aspect = torch.exp(
        _uniform(
            batch,
            tuple(math.log(value) for value in config.get("ratio", (0.85, 1.18))),
            device,
        )
    )
    crop_w = torch.sqrt(area * aspect).clamp(max=1.0)
    crop_h = torch.sqrt(area / aspect).clamp(max=1.0)
    angle = _uniform(batch, (-float(config.get("rotation", 10.0)),
                             float(config.get("rotation", 10.0))), device) * (math.pi / 180.0)
    translation = float(config.get("translation", 0.05))
    offset_x = (torch.rand(batch, device=device) * 2 - 1) * (1 - crop_w)
    offset_x += (torch.rand(batch, device=device) * 2 - 1) * translation * 2
    offset_y = (torch.rand(batch, device=device) * 2 - 1) * (1 - crop_h)
    offset_y += (torch.rand(batch, device=device) * 2 - 1) * translation * 2
    cosine, sine = torch.cos(angle), torch.sin(angle)
    flip = torch.ones(batch, device=device)
    if bool(config.get("horizontal_flip", False)):
        flip = torch.where(torch.rand(batch, device=device) < 0.5, -flip, flip)
    theta = torch.stack(
        (
            torch.stack((cosine * crop_w * flip, -sine * crop_h, offset_x), dim=1),
            torch.stack((sine * crop_w * flip, cosine * crop_h, offset_y), dim=1),
        ),
        dim=1,
    )
    grid = F.affine_grid(theta, (batch, 1, image_size, image_size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    contrast_range = float(config.get("contrast", 0.2))
    brightness_range = float(config.get("brightness", 0.1))
    contrast = 1 + _uniform(batch, (-contrast_range, contrast_range), device).view(batch, 1, 1, 1)
    brightness = _uniform(batch, (-brightness_range, brightness_range), device).view(batch, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean + brightness).clamp_(0, 1)

    blur_probability = float(config.get("blur_probability", 0.5))
    if blur_probability > 0:
        kernel_size = 9
        radius = kernel_size // 2
        sigma_bounds = tuple(config.get("blur_sigma", (0.1, 1.5)))
        blur_mask = torch.rand(batch, device=device) < blur_probability
        sigma = torch.where(
            blur_mask,
            _uniform(batch, sigma_bounds, device),
            torch.full((batch,), 1e-3, device=device),
        )
        positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-(positions[None, :] ** 2) / (2 * sigma[:, None] ** 2))
        kernel /= kernel.sum(dim=1, keepdim=True)
        x = x.reshape(1, batch, image_size, image_size)
        horizontal = kernel.view(batch, 1, 1, kernel_size)
        vertical = kernel.view(batch, 1, kernel_size, 1)
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=batch)
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), vertical, groups=batch)
        x = x.reshape(batch, 1, image_size, image_size)

    noise_probability = float(config.get("noise_probability", 0.5))
    noise_std = float(config.get("noise_std", 0.02))
    if noise_probability > 0 and noise_std > 0:
        mask = (torch.rand(batch, device=device) < noise_probability).view(batch, 1, 1, 1)
        std = torch.rand(batch, device=device).view(batch, 1, 1, 1) * noise_std * mask
        x = (x + torch.randn_like(x) * std).clamp_(0, 1)

    x = x.sub(0.5).div(0.5).expand(-1, 3, -1, -1)
    return x.contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
