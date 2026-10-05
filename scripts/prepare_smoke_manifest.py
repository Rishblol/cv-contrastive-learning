"""Create a small image-only manifest for checking the SimCLR pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/manifests/pretrain_train_leakage_free.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifests/pretrain_smoke.csv"),
    )
    parser.add_argument("--records", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.records < 1:
        raise ValueError("--records must be at least 1")
    if not args.source.is_file():
        raise FileNotFoundError(f"Prepared pretraining manifest not found: {args.source}")

    frame = pd.read_csv(args.source)
    if frame.empty:
        raise ValueError(f"Prepared pretraining manifest is empty: {args.source}")
    smoke_frame = frame.head(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    smoke_frame.to_csv(args.output, index=False)
    print(f"Wrote {len(smoke_frame)} records to {args.output}")


if __name__ == "__main__":
    main()
