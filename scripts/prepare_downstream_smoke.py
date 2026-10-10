"""Create a bounded patient-safe downstream smoke from a persisted training cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from chexpert_ssl.data import (
    assert_patient_disjoint,
    read_manifest,
    reject_final_partition,
    select_patient_cohort,
)
from chexpert_ssl.utils import file_hash, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/downstream/simclr_linear.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/smoke"))
    parser.add_argument("--train-patients", type=int, default=8)
    parser.add_argument("--development-patients", type=int, default=32)
    args = parser.parse_args()
    if min(args.train_patients, args.development_patients) < 1:
        parser.error("Smoke patient limits must be positive")
    config = load_config(args.config)
    train = read_manifest(config["train_manifest"])
    dev = read_manifest(config["development_manifest"])
    reject_final_partition(config["train_manifest"], train)
    reject_final_partition(config["development_manifest"], dev)
    assert_patient_disjoint(train, dev)
    train = select_patient_cohort(
        train, config["sampled_patients"], config["label_fraction"], config["seed"]
    )
    patients = sorted(train.patient_id.unique())[: args.train_patients]
    train = train.loc[train.patient_id.isin(patients)]
    dev = dev.loc[dev.patient_id.isin(sorted(dev.patient_id.unique())[: args.development_patients])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "downstream_train.csv", index=False)
    dev.to_csv(args.output_dir / "development.csv", index=False)
    save_json(
        args.output_dir / "cohort.json",
        {"seed": config["seed"], "label_fraction": 1.0, "patient_ids": patients},
    )
    save_json(
        args.output_dir / "provenance.json",
        {
            "source_cohort": config["sampled_patients"],
            "source_cohort_sha256": file_hash(config["sampled_patients"]),
            "train_records": len(train),
            "development_records": len(dev),
        },
    )


if __name__ == "__main__":
    main()
