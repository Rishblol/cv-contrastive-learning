"""Aggregate metric artifacts with explicit partition and initialization identity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chexpert_ssl.utils import load_config


def collect_results(root: Path, partition: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        payload = load_config(path)
        if payload.get("partition") != partition:
            continue
        metadata_path = path.parent / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_config(metadata_path)
        if metadata.get("status") != "complete":
            continue
        config = metadata["config"]
        training = config.get("training_config", config)
        mode = payload.get("mode", training.get("mode"))
        ssl_method = payload.get("ssl_method", training.get("ssl_method"))
        initialization = ssl_method or (
            "imagenet" if training.get("imagenet_pretrained") else "scratch"
        )
        method = f"{initialization}_{mode}"
        modality = training.get(
            "pretraining_modality",
            "image-only SSL"
            if ssl_method
            else "ImageNet"
            if training.get("imagenet_pretrained")
            else "none",
        )
        metrics = {
            key: value
            for key, value in payload.items()
            if key.startswith(
                (
                    "macro_",
                    "auroc_",
                    "auprc_",
                    "sensitivity_",
                    "specificity_",
                    "f1_",
                    "balanced_accuracy_",
                )
            )
            and isinstance(value, (float, int))
        }
        rows.append(
            {
                "run_id": payload.get("source_run_id", path.parent.name),
                "partition": partition,
                "method": method,
                "protocol": mode,
                "architecture": training.get("encoder"),
                "label_fraction": payload.get("label_fraction", training.get("label_fraction")),
                "seed": payload.get("seed", training.get("seed")),
                "pretraining_modality": modality,
                "metrics_path": str(path),
                **metrics,
            }
        )
    raw = pd.DataFrame(rows)
    if not raw.empty:
        identity = ["architecture", "method", "pretraining_modality", "label_fraction", "seed"]
        if raw.duplicated(identity, keep=False).any():
            raise ValueError(
                "Duplicate seed/method/budget results; select a single experiment root"
            )
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument(
        "--partition", choices=("development", "final_validation"), default="final_validation"
    )
    args = parser.parse_args()
    raw = collect_results(args.root, args.partition)
    output = args.output_dir / args.partition
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw_metrics.csv", index=False)
    if raw.empty:
        print(f"No completed {args.partition} metric artifacts found")
        return
    keys = ["partition", "architecture", "method", "pretraining_modality", "label_fraction"]
    metrics = [
        column
        for column in raw
        if column.startswith(
            (
                "macro_",
                "auroc_",
                "auprc_",
                "sensitivity_",
                "specificity_",
                "f1_",
                "balanced_accuracy_",
            )
        )
    ]
    summary = raw.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "label_efficiency_summary.csv", index=False)


if __name__ == "__main__":
    main()
