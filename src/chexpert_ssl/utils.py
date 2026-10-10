"""Configuration, reproducibility, and lightweight checkpoint utilities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@contextmanager
def artifact_lock(path: Path):
    """Hold an OS-backed lock; crashes release ownership without stale-lock cleanup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Another process owns this artifact: {path}") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another process owns this artifact: {path}") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def managed_run(function):
    """Reject concurrent writers and record failures after a run has started."""

    @wraps(function)
    def wrapped(config: dict):
        config = {
            **config,
            "implementation_sha256": implementation_hash(Path(function.__code__.co_filename)),
        }
        output_dir = Path(config["output_dir"])
        with artifact_lock(output_dir / ".run.lock"):
            metadata_path = output_dir / "run_metadata.json"
            previous = metadata_path.read_bytes() if metadata_path.exists() else None
            try:
                return function(config)
            except (Exception, KeyboardInterrupt) as error:
                if metadata_path.exists() and metadata_path.read_bytes() != previous:
                    metadata = load_config(metadata_path)
                    if metadata.get("status") == "running":
                        metadata.update(
                            status="interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "failed",
                            error=f"{type(error).__name__}: {error}",
                            ended_at_utc=datetime.now(timezone.utc).isoformat(),
                        )
                        save_json(metadata_path, metadata)
                raise

    return wrapped


def merge_config(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        result[key] = (
            merge_config(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def normalize_config(config: dict) -> dict:
    """Resolve numeric experiment settings consistently across YAML and JSON.

    YAML loaders can interpret scientific notation such as ``1e-05`` as a
    string. Normalize declared numeric keys rather than arbitrary text values.
    """
    float_keys = {
        "learning_rate",
        "encoder_learning_rate",
        "weight_decay",
        "label_fraction",
        "temperature",
        "momentum",
        "sinkhorn_epsilon",
        "dev_fraction",
    }
    integer_keys = {
        "seed",
        "batch_size",
        "epochs",
        "image_size",
        "num_workers",
        "prefetch_factor",
        "cache_workers",
        "cache_log_interval",
        "torch_num_threads",
        "early_stopping_patience",
        "bootstrap_samples",
        "hidden_dim",
        "projection_dim",
        "queue_size",
        "prototype_count",
        "prototype_freeze_steps",
    }
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value = normalize_config(value)
        elif isinstance(value, list):
            value = [normalize_config(item) if isinstance(item, dict) else item for item in value]
        if value is not None and key in float_keys:
            value = float(value)
        elif value is not None and key in integer_keys:
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{key} must be an integer")
            value = int(numeric)
        resolved[key] = value
    return resolved


def load_config(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    parents = config.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    resolved: dict[str, Any] = {}
    for parent in parents:
        resolved = merge_config(resolved, load_config(path.parent / parent, (*_seen, path)))
    return normalize_config(merge_config(resolved, config))


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash(entrypoint: Path) -> str:
    """Bind a run to the entry point and installed package sources, including uncommitted edits."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    payload = {path.name: file_hash(path) for path in files}
    payload["entrypoint"] = file_hash(entrypoint)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def config_signature(config: dict) -> str:
    payload = {
        key: value
        for key, value in normalize_config(config).items()
        if key not in {"output_dir", "resume"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate_training_config(config: dict, stage: str) -> None:
    from .data import TARGETS, UNCERTAIN_AS_POSITIVE
    from .ssl_methods import METHODS

    required = {
        "seed",
        "encoder",
        "image_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "output_dir",
        "num_workers",
        "resume",
        "target_policy",
        "optimizer",
        "precision",
        "multiprocessing_context",
    }
    required |= (
        {"manifest", "development_manifest", "ssl", "augmentation"}
        if stage == "pretrain"
        else {
            "train_manifest",
            "development_manifest",
            "sampled_patients",
            "mode",
            "label_fraction",
            "early_stopping_patience",
            "supervised_augmentation",
            "imagenet_pretrained",
        }
    )
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing {stage} configuration keys: {sorted(missing)}")
    if config["encoder"] not in config.get("architectures", ["resnet18", "resnet50"]):
        raise ValueError("Encoder is not allowed by the shared backbone configuration")
    expected = {label: int(label in UNCERTAIN_AS_POSITIVE) for label in TARGETS}
    if config["target_policy"] != expected:
        raise ValueError("Primary experiments require the declared U-Ones/U-Zeros target policy")
    for key in ("batch_size", "epochs", "image_size", "learning_rate"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["num_workers"]) < 0 or float(config["weight_decay"]) < 0:
        raise ValueError("num_workers and weight_decay must be nonnegative")
    if config["optimizer"] != "adamw" or config["precision"] not in {"auto", "float32"}:
        raise ValueError("Supported optimizer/precision: adamw and auto/float32")
    if config["multiprocessing_context"] not in {"spawn", "forkserver"}.intersection(
        multiprocessing.get_all_start_methods()
    ):
        raise ValueError("Use spawn or forkserver for safe training worker creation")
    if stage == "pretrain":
        if config.get("method") not in METHODS or config.get("schedule") != "cosine":
            raise ValueError("Pretraining requires a supported SSL method and cosine schedule")
        if int(config["batch_size"]) < 2:
            raise ValueError("SSL batch size must be at least two")
    else:
        if config["mode"] not in {
            "supervised",
            "ssl_linear",
            "ssl_finetune",
            "simclr_linear",
            "simclr_finetune",
        }:
            raise ValueError("Unsupported downstream mode")
        if (
            not 0 < float(config["label_fraction"]) <= 1
            or int(config["early_stopping_patience"]) < 1
        ):
            raise ValueError("Invalid label fraction or early-stopping patience")
        if not config["sampled_patients"]:
            raise ValueError("A persisted sampled_patients cohort is required")
        if config["mode"] != "supervised":
            if config.get("ssl_method") not in METHODS or not config.get("ssl_checkpoint"):
                raise ValueError("SSL transfer requires ssl_method and ssl_checkpoint")
            if config["imagenet_pretrained"]:
                raise ValueError("SSL and ImageNet initialization cannot be combined")
        if config["imagenet_pretrained"] and not config.get("imagenet_weights"):
            raise ValueError("ImageNet initialization requires an explicit weight variant")


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_metadata(config: dict[str, Any], command: list[str] | None = None) -> dict[str, Any]:
    """Common provenance persisted in every output directory."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "config": config,
        "command": command or sys.argv,
        "git_commit": commit,
        "python": sys.version,
        "torch": torch.__version__,
        "software_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "pandas",
                "Pillow",
                "PyYAML",
                "scikit-learn",
            )
        },
        "cuda_available": torch.cuda.is_available(),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_run_metadata(output_dir: Path, config: dict[str, Any]) -> None:
    path = output_dir / "run_metadata.json"
    metadata = run_metadata(config)
    if path.exists():
        previous = load_config(path)
        if config_signature(previous["config"]) != config_signature(config):
            raise ValueError("Output directory belongs to a different configuration")
        metadata["timestamp_utc"] = previous["timestamp_utc"]
        metadata["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "running"
    save_json(path, metadata)
    save_json(output_dir / "resolved_config.json", config)


def finish_run(output_dir: Path, **details: Any) -> None:
    path = output_dir / "run_metadata.json"
    metadata = load_config(path)
    metadata.update(details)
    metadata["status"] = "complete"
    metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(path, metadata)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
