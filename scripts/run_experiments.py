"""Materialize and execute the five-method image-only matrix in explicit stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from chexpert_ssl.experiments import build_plan
from chexpert_ssl.utils import file_hash, implementation_hash, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/label_efficiency.yaml")
    )
    parser.add_argument(
        "--stage",
        choices=("plan", "prepare", "smoke", "pretrain", "downstream", "evaluate", "aggregate"),
        default="plan",
    )
    parser.add_argument(
        "--methods", nargs="+", help="Restrict SSL methods or scratch/imagenet arms"
    )
    parser.add_argument("--budgets", type=float, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--augmentation-reviewed",
        action="store_true",
        help="Confirm anatomical review of the completed smoke augmentation preview",
    )
    args = parser.parse_args()
    matrix = load_config(args.config)
    plan = build_plan(matrix)
    available = set(matrix["pretrain_configs"]) | {"scratch", "imagenet"}
    if args.methods and not set(args.methods).issubset(available):
        parser.error("Unknown method filter")
    if args.budgets and not set(args.budgets).issubset(set(matrix["budgets"])):
        parser.error("Unknown budget filter")
    if args.seeds and not set(args.seeds).issubset(set(matrix["seeds"])):
        parser.error("Unknown seed filter")
    root = Path(matrix["generated_dir"])
    commands = {}
    for stage, configs in plan.items():
        commands[stage] = []
        for config in configs:
            run_id = Path(config["output_dir"]).name
            path = root / stage / f"{run_id}.json"
            if path.exists() and load_config(path) != config:
                raise ValueError(f"Generated plan changed; use a new generated_dir: {path}")
            save_json(path, config)
            script = "evaluate_checkpoint" if stage == "evaluate" else f"train_{stage}"
            commands[stage].append([sys.executable, f"scripts/{script}.py", "--config", str(path)])
    save_json(
        root / "plan.json",
        {"matrix": matrix, "counts": {k: len(v) for k, v in plan.items()}, "commands": commands},
    )
    print(
        f"Plan: {len(plan['pretrain'])} pretraining runs, {len(plan['downstream'])} downstream runs; {root}",
        flush=True,
    )
    if args.stage == "plan":
        return
    if args.stage == "prepare":
        subprocess.run(
            [sys.executable, "scripts/prepare_data.py", "--config", matrix["data_config"]],
            check=True,
        )
    elif args.stage == "smoke":
        for command in (
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "scripts/prepare_smoke_manifest.py"],
            [
                sys.executable,
                "scripts/train_pretrain.py",
                "--config",
                "configs/pretrain/simclr_smoke.yaml",
            ],
            [sys.executable, "scripts/prepare_downstream_smoke.py"],
            [
                sys.executable,
                "scripts/train_downstream.py",
                "--config",
                "configs/downstream/smoke.yaml",
            ],
        ):
            subprocess.run(command, check=True)
        print(
            "Inspect outputs/smoke/simclr-resnet18-gpu-cache/augmentation_pairs.png before full pretraining."
        )
    elif args.stage == "aggregate":
        subprocess.run(
            [sys.executable, "scripts/aggregate_results.py", "--partition", "final_validation"],
            check=True,
        )
    else:
        if args.stage == "pretrain":
            smoke = load_config(Path("configs/pretrain/simclr_smoke.yaml"))
            downstream_smoke = load_config(Path("configs/downstream/smoke.yaml"))
            for setting in (smoke, downstream_smoke):
                metadata = Path(setting["output_dir"]) / "run_metadata.json"
                if not metadata.exists() or load_config(metadata).get("status") != "complete":
                    raise ValueError("Complete the test/smoke stage before full pretraining")
                previous = load_config(metadata)["config"]
                entrypoint = "train_downstream.py" if "mode" in setting else "train_pretrain.py"
                if previous.get("implementation_sha256") != implementation_hash(
                    Path("scripts") / entrypoint
                ):
                    raise ValueError("Implementation changed; complete a new smoke run")
                if not all(previous.get(key) == value for key, value in setting.items()):
                    raise ValueError("Smoke configuration changed; complete a new smoke run")
                for key, digest in previous.get("provenance", {}).items():
                    if key in previous and file_hash(previous[key]) != digest:
                        raise ValueError("Smoke inputs changed; complete a new smoke run")
            if not args.augmentation_reviewed:
                raise ValueError(
                    "Inspect the smoke augmentation pairs, then pass --augmentation-reviewed"
                )
            preview = Path(smoke["output_dir"]) / "augmentation_pairs.png"
            save_json(
                root / "augmentation_review.json",
                {"preview": str(preview), "sha256": file_hash(preview)},
            )
        jobs = []
        for config, command in zip(plan[args.stage], commands[args.stage]):
            source = config
            if args.stage == "evaluate":
                source = next(
                    c
                    for c in plan["downstream"]
                    if str(Path(c["output_dir"]) / "best.pt") == config["checkpoint"]
                )
            method = (
                source.get("ssl_method")
                or source.get("method")
                or ("imagenet" if source.get("imagenet_pretrained") else "scratch")
            )
            if args.methods and method not in args.methods:
                continue
            if args.stage != "pretrain":
                if args.budgets and source["label_fraction"] not in args.budgets:
                    continue
                if args.seeds and source["seed"] not in args.seeds:
                    continue
            jobs.append((config, command))
        # Check every requested selection lock before any official validation is opened.
        if args.stage == "evaluate":
            import torch

            from chexpert_ssl.selection import verify_selection

            for config, _ in jobs:
                checkpoint = torch.load(
                    config["checkpoint"], map_location="cpu", weights_only=False
                )
                verify_selection(config, checkpoint)
        for config, command in jobs:
            metadata = Path(config["output_dir"]) / "run_metadata.json"
            if metadata.exists() and load_config(metadata).get("status") == "complete":
                previous = load_config(metadata)["config"]
                if previous.get("implementation_sha256") != implementation_hash(Path(command[1])):
                    raise ValueError(f"Completed run implementation changed: {metadata}")
                if not all(previous.get(key) == value for key, value in config.items()):
                    raise ValueError(f"Completed run differs from plan: {metadata}")
                for key, digest in previous.get("provenance", {}).items():
                    if key in previous and file_hash(previous[key]) != digest:
                        raise ValueError(f"Completed run input changed: {previous[key]}")
                for name in (
                    "metrics.json",
                    "best.pt" if args.stage != "evaluate" else "predictions.csv",
                ):
                    if not (Path(config["output_dir"]) / name).exists():
                        raise ValueError(f"Completed run is missing {name}: {metadata}")
                print(f"Reusing completed run: {config['output_dir']}", flush=True)
                continue
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
