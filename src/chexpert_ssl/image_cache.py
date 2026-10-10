"""Disk-backed grayscale image cache for repeated SSL training epochs."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from chexpert_ssl.data import manifest_hash

CACHE_VERSION = 1


class CachedGrayscaleDataset(Dataset):
    """Read resized uint8 grayscale images from a memory-mapped NumPy cache."""

    def __init__(self, cache_path: Path, image_count: int) -> None:
        self.cache_path = Path(cache_path)
        self.image_count = image_count
        self._images: np.ndarray | None = None

    def __len__(self) -> int:
        return self.image_count

    def __getstate__(self) -> dict:
        # Spawned workers reopen the mmap instead of serializing the entire cache.
        return {**self.__dict__, "_images": None}

    def __getitem__(self, index: int) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.cache_path, mmap_mode="r")
        return np.array(self._images[index], dtype=np.uint8, copy=True)


def _decode_image(index: int, image_path: str, resolution: int) -> tuple[int, np.ndarray]:
    with Image.open(image_path) as image:
        resized = image.convert("L").resize(
            (resolution, resolution), resample=Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.uint8)
    return index, pixels


def ensure_grayscale_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    resolution: int,
    workers: int = 16,
    log_interval: int = 10_000,
) -> tuple[CachedGrayscaleDataset, bool]:
    """Build or reuse an image cache keyed by manifest contents and resolution."""
    if "image_path" not in frame:
        raise ValueError("Pretraining manifest must contain an image_path column")
    if resolution < 32:
        raise ValueError("Cache resolution must be at least 32 pixels")
    if workers < 1:
        raise ValueError("Cache workers must be at least 1")

    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    signature = {
        "cache_version": CACHE_VERSION,
        "resolution": resolution,
        "records": len(frame),
        "manifest_hash": manifest_hash(frame),
    }
    if cache_path.is_file() and metadata_path.is_file():
        try:
            metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path, mmap_mode="r")
            if (
                metadata == signature
                and cached.shape == (len(frame), resolution, resolution)
                and cached.dtype == np.uint8
            ):
                print(f"Reusing grayscale image cache: {cache_path}", flush=True)
                return CachedGrayscaleDataset(cache_path, len(frame)), False
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = len(frame) * resolution * resolution + 1_048_576
    free_bytes = shutil.disk_usage(cache_path.parent).free
    if free_bytes < required_bytes:
        required_gib = required_bytes / 2**30
        free_gib = free_bytes / 2**30
        raise OSError(
            f"Image cache needs about {required_gib:.2f} GiB but only "
            f"{free_gib:.2f} GiB is free at {cache_path.parent}"
        )
    temporary_path = cache_path.with_name(f"{cache_path.stem}.building.npy")
    images: np.memmap | None = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(frame), resolution, resolution),
    )
    jobs = iter((index, str(path), resolution) for index, path in enumerate(frame["image_path"]))
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(frame)))) as executor:
            while batch := list(islice(jobs, 2048)):
                for index, pixels in executor.map(lambda job: _decode_image(*job), batch):
                    images[index] = pixels
                    completed += 1
                    if completed % log_interval == 0 or completed == len(frame):
                        print(f"cache images={completed}/{len(frame)}", flush=True)
        images.flush()
        del images
        images = None
        temporary_path.replace(cache_path)
        metadata_temporary = metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        metadata_temporary.replace(metadata_path)
    except Exception:
        if images is not None:
            del images
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Built grayscale image cache: {cache_path}", flush=True)
    return CachedGrayscaleDataset(cache_path, len(frame)), True
