"""CheXpert manifests, patient-safe splits, datasets, and image transforms."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from hashlib import sha256
from pathlib import Path

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


def patient_identity(identifier: str) -> str:
    """Compare decimal patient IDs independently of legacy zero-padding loss."""
    value = str(identifier)
    return value.lstrip("0") or "0" if re.fullmatch(r"[0-9]+", value) else value


def read_manifest(path: str | Path, *, labeled: bool = True) -> pd.DataFrame:
    """Load prepared records without losing zero-padded patient/study identifiers."""
    frame = pd.read_csv(path, dtype={"patient_id": "string", "study_id": "string"})
    required = {"patient_id", "image_path"}
    if labeled:
        required.update(target_columns())
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Invalid or empty manifest {path}; missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"Manifest {path} contains missing identifiers, paths, or targets")
    if labeled and not frame[target_columns()].isin([0, 1]).all().all():
        raise ValueError(f"Manifest {path} must contain binary resolved targets")
    if labeled:
        for label in TARGETS:
            if label in frame:
                expected = frame[label].map(
                    lambda value, target=label: resolve_target(value, target)
                )
                if not expected.eq(frame[f"target_{label}"]).all():
                    raise ValueError(
                        f"Resolved {label} targets do not match the primary uncertainty policy"
                    )
    if "Path" in frame:
        expected = frame["Path"].map(
            lambda value: extract_identifier(value, _PATIENT_RE, "patient ID")
        )
        equivalent = expected.map(patient_identity).eq(frame.patient_id.map(patient_identity))
        if not equivalent.all():
            raise ValueError(f"Manifest {path} has patient IDs inconsistent with source paths")
        restored = int((~expected.eq(frame.patient_id)).sum())
        frame["patient_id"] = expected.astype("string")
        frame.attrs["patient_id_padding_restored_rows"] = restored
    return frame


def reject_final_partition(path: str | Path, frame: pd.DataFrame) -> None:
    """Prevent official-validation records from entering a training/selection stage."""
    if Path(path).name in {"valid.csv", "final_validation.csv"}:
        raise ValueError("Official validation cannot be used for training or development selection")
    paths = frame.image_path.astype(str).str.replace("\\", "/", regex=False)
    if paths.str.contains(r"(?:^|/)valid/", regex=True).any():
        raise ValueError(
            "Official validation images cannot be used for training or development selection"
        )


def select_patient_cohort(
    frame: pd.DataFrame, path: str | Path, fraction: float, seed: int
) -> pd.DataFrame:
    """Require and validate the persisted patient cohort shared by all methods."""
    cohort = json.loads(Path(path).read_text(encoding="utf-8"))
    if cohort.get("seed") != seed or cohort.get("label_fraction") != fraction:
        raise ValueError("Cohort seed/budget does not match the run configuration")
    identifiers = cohort.get("patient_ids", [])
    if not identifiers or any(not isinstance(value, str) for value in identifiers):
        raise ValueError("Cohort must contain nonempty string patient IDs")
    # Source paths supply the canonical spelling. Resolve legacy cohort IDs by
    # numeric identity without changing their membership, order, or file contents.
    available = {}
    for identifier in frame.patient_id.unique():
        identity = patient_identity(identifier)
        if identity in available and available[identity] != identifier:
            raise ValueError("Manifest contains ambiguous zero-padded patient IDs")
        available[identity] = identifier
    identities = [patient_identity(identifier) for identifier in identifiers]
    if len(set(identities)) != len(identifiers) or not set(identities).issubset(available):
        raise ValueError("Cohort contains duplicate or unknown patient IDs")
    patients = {available[identity] for identity in identities}
    expected_count = max(1, round(frame.patient_id.nunique() * fraction))
    if len(patients) != expected_count:
        raise ValueError("Cohort patient count does not match the configured budget")
    selected = frame.loc[frame.patient_id.isin(patients)].reset_index(drop=True)
    selected.attrs["cohort_patient_id_padding_restored"] = sum(
        identifier != available[identity]
        for identifier, identity in zip(identifiers, identities, strict=True)
    )
    return selected


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


def prepare_manifest(
    csv_path: Path, dataset_root: Path, frontal_only: bool = False
) -> pd.DataFrame:
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
        frame[f"target_{target}"] = frame[target].map(
            lambda value, label=target: resolve_target(value, label)
        )
    return frame.reset_index(drop=True)


def assert_patient_disjoint(*frames: pd.DataFrame) -> None:
    """Raise when any two supplied manifests contain the same patient."""
    patient_sets = [set(frame["patient_id"].map(patient_identity)) for frame in frames]
    for index, current in enumerate(patient_sets):
        for other_index, other in enumerate(patient_sets[index + 1 :], start=index + 1):
            overlap = current.intersection(other)
            if overlap:
                raise ValueError(
                    f"Patient leakage between partitions {index} and {other_index}: "
                    f"{len(overlap)} shared patients"
                )


def exclude_development_patients(
    pretrain_manifest: pd.DataFrame, development_manifest: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Remove development patients from an existing pretraining manifest."""
    for label, frame in (("pretraining", pretrain_manifest), ("development", development_manifest)):
        if "patient_id" not in frame.columns:
            raise ValueError(f"{label} manifest must contain a patient_id column")
    development_ids = set(development_manifest["patient_id"].astype(str))
    mask = ~pretrain_manifest["patient_id"].astype(str).isin(development_ids)
    filtered = pretrain_manifest.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No pretraining rows remain after excluding development patients")
    assert_patient_disjoint(filtered, development_manifest)
    removed = pretrain_manifest.loc[~mask, "patient_id"].astype(str).nunique()
    return filtered, int(removed)


def split_by_patient(
    frame: pd.DataFrame, dev_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic patient-disjoint development partition.

    Patients are assigned using a shuffled order, rather than individual studies,
    so the split remains safe even when a patient has multiple studies or views.
    Pre/post-split prevalence is recorded by ``prepare_data.py``.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    patients = np.array(sorted(frame["patient_id"].astype(str).unique()))
    if len(patients) < 2:
        raise ValueError("At least two patients are required for a train/development split")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    dev_count = min(len(patients) - 1, max(1, round(len(patients) * dev_fraction)))
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
    count = max(1, round(len(patients) * fraction))
    chosen = set(patients[:count])
    return frame.loc[frame["patient_id"].astype(str).isin(chosen)].reset_index(drop=True)


def nested_patient_ids(
    frame: pd.DataFrame, fractions: Iterable[float], seed: int
) -> dict[float, list[str]]:
    """Return nested deterministic patient samples for every requested budget."""
    patients = np.array(sorted(frame["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    result: dict[float, list[str]] = {}
    for fraction in sorted({float(value) for value in fractions}):
        if not 0 < fraction <= 1:
            raise ValueError("All fractions must be in (0, 1]")
        count = max(1, round(len(patients) * fraction))
        result[fraction] = patients[:count].tolist()
    return result


def manifest_hash(frame: pd.DataFrame) -> str:
    """Stable content hash used to tie artifacts back to their source manifest."""
    columns = [
        column
        for column in ("Path", "image_path", "patient_id", *target_columns())
        if column in frame
    ]
    payload = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return sha256(payload).hexdigest()


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


def supervised_transform(
    image_size: int, train: bool, config: dict | None = None
) -> transforms.Compose:
    config = config or {}
    operations: list[Callable] = [transforms.Resize((image_size, image_size))]
    if train:
        operations.append(
            transforms.RandomAffine(
                degrees=float(config.get("rotation", 5)),
                translate=tuple(config.get("translation", (0.02, 0.02))),
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(operations)


def simclr_transform(image_size: int, horizontal_flip: bool = False) -> transforms.Compose:
    """Conservative radiograph augmentation policy; no hue/saturation or flip."""
    operations: list[Callable] = [
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    if horizontal_flip:
        operations.insert(2, transforms.RandomHorizontalFlip())
    return transforms.Compose(operations)


def existing_images(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return decodable images and a data-quality report for invalid records."""
    reasons: list[str] = []
    for value in frame["image_path"]:
        path = Path(value)
        if not path.is_file():
            reasons.append("missing")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            reasons.append("")
        except (OSError, ValueError):
            reasons.append("unreadable")
    report = frame.loc[[bool(reason) for reason in reasons]].copy()
    report["data_quality_reason"] = [reason for reason in reasons if reason]
    valid = frame.loc[[not bool(reason) for reason in reasons]].copy()
    return valid.reset_index(drop=True), report.reset_index(drop=True)


def prevalence(frame: pd.DataFrame) -> dict[str, float]:
    """Resolved target prevalence, including zero prevalence labels."""
    return {target: float(frame[f"target_{target}"].mean()) for target in TARGETS}
