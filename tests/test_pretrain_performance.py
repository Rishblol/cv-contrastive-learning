from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image

from chexpert_ssl.gpu_augment import augment_grayscale_batch
from chexpert_ssl.image_cache import ensure_grayscale_cache


def test_grayscale_gpu_augmentation_returns_finite_normalized_rgb() -> None:
    images = torch.randint(0, 256, (4, 32, 32), dtype=torch.uint8)
    augmented = augment_grayscale_batch(
        images,
        32,
        {
            "scale": (0.8, 1.0),
            "ratio": (0.9, 1.1),
            "blur_probability": 0.0,
            "noise_probability": 0.0,
        },
        channels_last=False,
    )

    assert augmented.shape == (4, 3, 32, 32)
    assert augmented.min() >= -1.0
    assert augmented.max() <= 1.0
    assert torch.isfinite(augmented).all()


def test_grayscale_cache_is_built_then_reused(tmp_path) -> None:
    image_paths = []
    for index in range(3):
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((40, 50), index * 50, dtype=np.uint8)).save(path)
        image_paths.append(str(path))
    manifest = pd.DataFrame(
        {"Path": image_paths, "image_path": image_paths, "patient_id": ["1", "2", "3"]}
    )
    cache_path = tmp_path / "cache" / "images.npy"

    dataset, built = ensure_grayscale_cache(manifest, cache_path, resolution=32, workers=2)
    first_image = dataset[1]
    _, reused = ensure_grayscale_cache(manifest, cache_path, resolution=32, workers=2)

    assert built
    assert not reused
    assert len(dataset) == 3
    assert first_image.shape == (32, 32)
    assert first_image.dtype == np.uint8
