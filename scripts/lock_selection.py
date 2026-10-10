"""Lock a completed run after reviewing its development results."""

from __future__ import annotations

import argparse
from pathlib import Path

from chexpert_ssl.selection import lock_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reason", required=True, help="Decision based exclusively on development results"
    )
    args = parser.parse_args()
    print(lock_run(args.run_dir, args.reason))


if __name__ == "__main__":
    main()
