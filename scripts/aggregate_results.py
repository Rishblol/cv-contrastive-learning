"""Aggregate raw metric files into seed summaries and label-efficiency plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()
    rows = []
    for path in args.root.rglob("metrics.json"):
        if args.output_dir in path.parents:
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata_path = path.parent / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        config = metadata.get("config", {})
        if "best_dev_auroc" not in payload and "macro_auroc" not in payload:
            continue
        mode = payload.get("mode", config.get("mode", "unknown"))
        ssl_method = payload.get("ssl_method", config.get("ssl_method"))
        method = f"{ssl_method}_{mode}" if ssl_method else mode
        imagenet_pretrained = bool(config.get("imagenet_pretrained", False))
        if imagenet_pretrained:
            modality = "ImageNet + image-only SSL" if ssl_method else "ImageNet"
        else:
            modality = config.get(
                "pretraining_modality",
                "image-only SSL" if ssl_method else "none",
            )
        rows.append({
            "run_id": path.parent.name,
            "method": method,
            "protocol": mode,
            "architecture": config.get("encoder", "unknown"),
            "label_fraction": payload.get("label_fraction", config.get("label_fraction")),
            "seed": config.get("seed"),
            "pretraining_modality": modality,
            "macro_auroc": payload.get("macro_auroc", payload.get("best_dev_auroc")),
            "macro_auprc": payload.get("macro_auprc"),
            "metrics_path": str(path),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "raw_metrics.csv", index=False)
    if raw.empty:
        return
    summary = raw.groupby(["architecture", "method", "pretraining_modality", "label_fraction"], dropna=False).agg(
        macro_auroc_mean=("macro_auroc", "mean"), macro_auroc_std=("macro_auroc", "std"),
        macro_auprc_mean=("macro_auprc", "mean"), macro_auprc_std=("macro_auprc", "std"), runs=("run_id", "count")
    ).reset_index()
    summary.to_csv(args.output_dir / "label_efficiency_summary.csv", index=False)


if __name__ == "__main__":
    main()
