"""Create portable CheXpert manifests and patient-disjoint development splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from chexpert_ssl.data import assert_patient_disjoint, existing_images, prepare_manifest, split_by_patient
from chexpert_ssl.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/manifests"))
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frontal-only-downstream", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    train = prepare_manifest(root / "train.csv", root, frontal_only=False)
    valid = prepare_manifest(root / "valid.csv", root, frontal_only=args.frontal_only_downstream)
    train, missing_train = existing_images(train)
    valid, missing_valid = existing_images(valid)
    downstream_source = train
    if args.frontal_only_downstream:
        downstream_source = train.loc[train["Frontal/Lateral"].eq("Frontal")].copy()
    downstream_train, dev = split_by_patient(downstream_source, args.dev_fraction, args.seed)
    assert_patient_disjoint(downstream_train, dev, valid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "pretrain_train.csv", index=False)
    downstream_train.to_csv(args.output_dir / "downstream_train.csv", index=False)
    dev.to_csv(args.output_dir / "development.csv", index=False)
    valid.to_csv(args.output_dir / "final_validation.csv", index=False)
    missing_train.to_csv(args.output_dir / "missing_pretrain_images.csv", index=False)
    missing_valid.to_csv(args.output_dir / "missing_validation_images.csv", index=False)
    save_json(
        args.output_dir / "data_report.json",
        {
            "dataset_root": str(root),
            "pretrain_records": len(train),
            "downstream_train_records": len(downstream_train),
            "development_records": len(dev),
            "final_validation_records": len(valid),
            "missing_pretrain_images": len(missing_train),
            "missing_validation_images": len(missing_valid),
            "dev_fraction": args.dev_fraction,
            "seed": args.seed,
            "frontal_only_downstream": args.frontal_only_downstream,
        },
    )
    print(f"Prepared manifests in {args.output_dir}")


if __name__ == "__main__":
    main()
