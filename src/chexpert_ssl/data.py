"""CheXpert manifests, patient-safe splits, datasets, and image transforms."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TARGETS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
UNCERTAIN_AS_POSITIVE = frozenset({"Atelectasis", "Edema"})
FRONTAL_VIEWS = frozenset({"AP", "PA"})

_PATIENT_RE = re.compile(r"patient(\d+)", re.IGNORECASE)
_STUDY_RE = re.compile(r"study(\d+)", re.IGNORECASE)


def resolve_target(value: object, target: str) -> int:
    """Apply the declared CheXpert U-Ones/U-Zeros policy to one raw label."""
    if pd.isna(value):
        return 0
    value = int(value)
    if value == -1:
        return int(target in UNCERTAIN_AS_POSITIVE)
    if value not in (0, 1):
        raise ValueError(f"Unexpected label value {value!r} for {target}")
    return value


def local_image_path(raw_path: str, dataset_root: Path) -> Path:
    """Map a CheXpert CSV path to a portable path rooted at ``dataset_root``."""
    parts = Path(str(raw_path).replace("\\", "/")).parts
    root_name = dataset_root.name
    try:
        index = parts.index(root_name)
        relative = Path(*parts[index + 1 :])
    except ValueError:
        relative = Path(*parts[-3:]) if len(parts) >= 3 else Path(*parts)
    return dataset_root / relative


def extract_identifier(raw_path: str, pattern: re.Pattern[str], label: str) -> str:
    match = pattern.search(str(raw_path))
    if not match:
        raise ValueError(f"Could not extract {label} from {raw_path!r}")
    return match.group(1)


def prepare_manifest(csv_path: Path, dataset_root: Path, frontal_only: bool = False) -> pd.DataFrame:
    """Load a CheXpert manifest and add portable paths, IDs, and resolved targets."""
    frame = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CheXpert columns: {sorted(missing)}")

    if frontal_only:
        frame = frame.loc[frame["Frontal/Lateral"].eq("Frontal")].copy()
    else:
        frame = frame.copy()

    frame["image_path"] = frame["Path"].map(lambda path: str(local_image_path(path, dataset_root)))
    frame["patient_id"] = frame["Path"].map(
        lambda path: extract_identifier(path, _PATIENT_RE, "patient ID")
    )
    frame["study_id"] = frame["Path"].map(
        lambda path: extract_identifier(path, _STUDY_RE, "study ID")
    )
    for target in TARGETS:
        frame[f"target_{target}"] = frame[target].map(lambda value: resolve_target(value, target))
    return frame.reset_index(drop=True)


def assert_patient_disjoint(*frames: pd.DataFrame) -> None:
    """Raise when any two supplied manifests contain the same patient."""
    patient_sets = [set(frame["patient_id"].astype(str)) for frame in frames]
    for index, current in enumerate(patient_sets):
        for other_index, other in enumerate(patient_sets[index + 1 :], start=index + 1):
            overlap = current.intersection(other)
            if overlap:
                raise ValueError(
                    f"Patient leakage between partitions {index} and {other_index}: "
                    f"{len(overlap)} shared patients"
                )


def split_by_patient(frame: pd.DataFrame, dev_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic patient-disjoint development partition."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    patients = np.array(sorted(frame["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    dev_count = max(1, int(round(len(patients) * dev_fraction)))
    dev_patients = set(patients[:dev_count])
    dev = frame.loc[frame["patient_id"].astype(str).isin(dev_patients)].copy()
    train = frame.loc[~frame["patient_id"].astype(str).isin(dev_patients)].copy()
    assert_patient_disjoint(train, dev)
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def sample_patient_budget(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Sample a deterministic fraction of patients for one labeled-data budget."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    patients = np.array(sorted(frame["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    count = max(1, int(round(len(patients) * fraction)))
    chosen = set(patients[:count])
    return frame.loc[frame["patient_id"].astype(str).isin(chosen)].reset_index(drop=True)


def target_columns() -> list[str]:
    return [f"target_{target}" for target in TARGETS]


class CheXpertDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Image dataset that returns an X-ray and its five resolved targets."""

    def __init__(self, frame: pd.DataFrame, transform: Callable | None = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.columns = target_columns()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        targets = torch.tensor(row[self.columns].to_numpy(dtype=np.float32), dtype=torch.float32)
        return image, targets


class SimCLRDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset returning two independently augmented views of each image."""

    def __init__(self, frame: pd.DataFrame, transform: Callable) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.frame.iloc[index]["image_path"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), self.transform(image)


def supervised_transform(image_size: int, train: bool) -> transforms.Compose:
    operations: list[Callable] = [transforms.Resize((image_size, image_size))]
    if train:
        operations.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(operations)


def simclr_transform(image_size: int) -> transforms.Compose:
    """Conservative radiograph augmentation policy; no hue/saturation or flip."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def existing_images(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return readable-path candidates and a report frame for missing paths."""
    exists = frame["image_path"].map(lambda value: Path(value).is_file())
    return frame.loc[exists].reset_index(drop=True), frame.loc[~exists].reset_index(drop=True)
