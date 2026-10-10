"""Development-only selection records bound to immutable checkpoint artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch

from .utils import artifact_lock, config_signature, file_hash, load_config, save_json


def lock_run(run_dir: Path, reason: str) -> Path:
    """Record a human selection decision without opening held-out records."""
    with artifact_lock(run_dir / ".run.lock"):
        return _lock_run(run_dir, reason)


def _lock_run(run_dir: Path, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("A development-selection reason is required")
    metadata = load_config(run_dir / "run_metadata.json")
    metrics = load_config(run_dir / "metrics.json")
    if metadata.get("status") != "complete" or metrics.get("partition") != "development":
        raise ValueError("Only completed development-selected runs can be locked")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    signature = checkpoint.get("training_signature")
    if (
        not signature
        or signature != config_signature(metadata["config"])
        or signature != metrics.get("training_signature")
    ):
        raise ValueError("Checkpoint, metrics, and metadata configurations disagree")
    thresholds = load_config(run_dir / "thresholds.json")
    if thresholds != checkpoint.get("thresholds"):
        raise ValueError("Thresholds do not belong to the selected checkpoint")
    lock = {
        "selected_on": "development",
        "reason": reason,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_dir.name,
        "training_signature": signature,
        "checkpoint_sha256": file_hash(run_dir / "best.pt"),
        "thresholds_sha256": file_hash(run_dir / "thresholds.json"),
        "metrics_sha256": file_hash(run_dir / "metrics.json"),
    }
    path = run_dir / "selection_lock.json"
    if path.exists():
        raise ValueError("Selection lock already exists; selected runs are immutable")
    save_json(path, lock)
    return path


def verify_selection(config: dict, checkpoint: dict) -> tuple[dict, dict]:
    lock = load_config(Path(config["selection_lock"]))
    training = checkpoint["config"]
    signature = checkpoint.get("training_signature")
    if (
        lock.get("selected_on") != "development"
        or not signature
        or signature != config_signature(training)
    ):
        raise ValueError("A valid development-selection lock is required")
    if lock.get("training_signature") != signature:
        raise ValueError("Selection lock configuration does not match checkpoint")
    for config_key, lock_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("thresholds", "thresholds_sha256"),
    ):
        if file_hash(config[config_key]) != lock.get(lock_key):
            raise ValueError(f"Selected {config_key} artifact changed after locking")
    metrics_path = Path(config["checkpoint"]).parent / "metrics.json"
    if file_hash(metrics_path) != lock.get("metrics_sha256"):
        raise ValueError("Selected development metrics changed after locking")
    for key in ("encoder", "image_size", "target_policy", "supervised_augmentation"):
        if key in config and config[key] != training[key]:
            raise ValueError(f"Evaluation {key} conflicts with selected training configuration")
    thresholds = load_config(Path(config["thresholds"]))
    if thresholds != checkpoint.get("thresholds"):
        raise ValueError("Thresholds do not match selected checkpoint")
    return training, lock
