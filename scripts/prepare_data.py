"""Create portable CheXpert manifests and patient-disjoint development splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from chexpert_ssl.data import (
    assert_patient_disjoint,
    existing_images,
    manifest_hash,
    nested_patient_ids,
    prepare_manifest,
    prevalence,
    split_by_patient,
)
from chexpert_ssl.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/manifests"))
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frontal-only-downstream", action="store_true")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 1.0])
    parser.add_argument("--budget-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly regenerate all manifests and cohorts"
    )
    initial, _ = parser.parse_known_args()
    if initial.config:
        config = load_config(initial.config)
        unknown = set(config).difference({action.dest for action in parser._actions})
        if unknown:
            parser.error(f"Unknown data configuration keys: {sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root or a data configuration is required")
    if not 0 < args.dev_fraction < 1 or any(not 0 < fraction <= 1 for fraction in args.budgets):
        parser.error("Invalid development fraction or label budget")
    return args


def main() -> None:
    args = parse_args()
    if not args.overwrite and (args.output_dir / "data_report.json").exists():
        raise ValueError(
            "Prepared data already exists; reuse the cohorts or explicitly pass --overwrite"
        )
    root = args.dataset_root
    train = prepare_manifest(root / "train.csv", root, frontal_only=False)
    valid = prepare_manifest(root / "valid.csv", root, frontal_only=args.frontal_only_downstream)
    train, missing_train = existing_images(train)
    valid, missing_valid = existing_images(valid)
    pretrain_train, development_source = split_by_patient(train, args.dev_fraction, args.seed)
    downstream_train, dev = pretrain_train, development_source
    if args.frontal_only_downstream:
        downstream_train = pretrain_train.loc[
            pretrain_train["Frontal/Lateral"].eq("Frontal")
        ].copy()
        dev = development_source.loc[development_source["Frontal/Lateral"].eq("Frontal")].copy()
    if downstream_train.empty or dev.empty:
        raise ValueError("View filtering produced an empty train/development partition")
    assert_patient_disjoint(downstream_train, dev, valid)
    assert_patient_disjoint(pretrain_train, development_source, valid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.output_dir.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    pretrain_train.to_csv(args.output_dir / "pretrain_train.csv", index=False)
    pretrain_train.to_csv(args.output_dir / "pretrain_train_leakage_free.csv", index=False)
    downstream_train.to_csv(args.output_dir / "downstream_train.csv", index=False)
    dev.to_csv(args.output_dir / "development.csv", index=False)
    valid.to_csv(args.output_dir / "final_validation.csv", index=False)
    missing_train.to_csv(args.output_dir / "missing_pretrain_images.csv", index=False)
    missing_valid.to_csv(args.output_dir / "missing_validation_images.csv", index=False)
    pretrain_train[["patient_id"]].drop_duplicates().to_csv(
        splits_dir / "train_patients.csv", index=False
    )
    development_source[["patient_id"]].drop_duplicates().to_csv(
        splits_dir / "development_patients.csv", index=False
    )
    valid[["patient_id"]].drop_duplicates().to_csv(
        splits_dir / "final_validation_patients.csv", index=False
    )
    for budget_seed in args.budget_seeds:
        for fraction, patients in nested_patient_ids(
            downstream_train, args.budgets, budget_seed
        ).items():
            label = f"{fraction:g}".replace(".", "p")
            save_json(
                splits_dir / f"budget_{label}_seed_{budget_seed}.json",
                {"seed": budget_seed, "label_fraction": fraction, "patient_ids": patients},
            )
    save_json(
        args.output_dir / "data_report.json",
        {
            "dataset_root": str(root),
            "pretrain_records": len(pretrain_train),
            "downstream_train_records": len(downstream_train),
            "development_records": len(dev),
            "final_validation_records": len(valid),
            "missing_pretrain_images": len(missing_train),
            "missing_validation_images": len(missing_valid),
            "dev_fraction": args.dev_fraction,
            "seed": args.seed,
            "frontal_only_downstream": args.frontal_only_downstream,
            "downstream_filtered_records": len(train) - len(downstream_train) - len(dev),
            "manifest_hashes": {
                "pretrain_train": manifest_hash(pretrain_train),
                "downstream_train": manifest_hash(downstream_train),
                "development": manifest_hash(dev),
                "final_validation": manifest_hash(valid),
            },
            "prevalence": {
                "downstream_train": prevalence(downstream_train),
                "development": prevalence(dev),
                "final_validation": prevalence(valid),
            },
            "uncertainty_as_positive": ["Atelectasis", "Edema"],
            "budget_seeds": args.budget_seeds,
            "budgets": args.budgets,
        },
    )
    print(f"Prepared manifests in {args.output_dir}")


if __name__ == "__main__":
    main()
