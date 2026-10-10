"""Create a development-patient-free pretraining manifest without reading images."""

from __future__ import annotations

import argparse
from pathlib import Path

from chexpert_ssl.data import (
    exclude_development_patients,
    manifest_hash,
    read_manifest,
    reject_final_partition,
)
from chexpert_ssl.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-manifest",
        type=Path,
        default=Path("data/processed/manifests/pretrain_train.csv"),
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/processed/manifests/development.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifests/pretrain_train_leakage_free.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for source in (args.pretrain_manifest, args.development_manifest):
        if not source.is_file():
            raise FileNotFoundError(f"Manifest not found: {source}")
    if args.output.resolve() in {
        args.pretrain_manifest.resolve(),
        args.development_manifest.resolve(),
    }:
        raise ValueError("Output must be a separate file; source manifests will not be overwritten")

    pretrain = read_manifest(args.pretrain_manifest, labeled=False)
    development = read_manifest(args.development_manifest, labeled=False)
    reject_final_partition(args.pretrain_manifest, pretrain)
    reject_final_partition(args.development_manifest, development)
    repaired, removed_patients = exclude_development_patients(pretrain, development)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(args.output, index=False)
    report_path = args.output.with_name(f"{args.output.stem}_repair_report.json")
    save_json(
        report_path,
        {
            "source_manifest": str(args.pretrain_manifest),
            "development_manifest": str(args.development_manifest),
            "output_manifest": str(args.output),
            "source_records": len(pretrain),
            "output_records": len(repaired),
            "removed_records": len(pretrain) - len(repaired),
            "removed_development_patients": removed_patients,
            "source_manifest_hash": manifest_hash(pretrain),
            "output_manifest_hash": manifest_hash(repaired),
        },
    )
    print(
        f"Wrote {len(repaired):,} pretraining records to {args.output}; "
        f"excluded {removed_patients:,} development patients ({len(pretrain) - len(repaired):,} records). "
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
