"""Configuration, reproducibility, and lightweight checkpoint utilities."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


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
        "cuda_available": torch.cuda.is_available(),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_run_metadata(output_dir: Path, config: dict[str, Any]) -> None:
    save_json(output_dir / "run_metadata.json", run_metadata(config))


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
