"""Materialize and execute the five-method image-only matrix in explicit stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from chexpert_ssl.experiments import build_plan
from chexpert_ssl.utils import (
    config_signature,
    file_hash,
    implementation_hash,
    load_config,
    save_json,
)


def smoke_settings(matrix: dict) -> tuple[dict, dict, Path, str]:
    """Version smoke outputs by their settings, source inputs, and implementation."""
    if matrix.get("notebook_smoke"):
        suite, root, source, _ = notebook_smoke_suite(matrix)
        return suite["pretrain-simclr"], suite["downstream-linear"], root, source["manifest"]
    pretrain = load_config(Path("configs/pretrain/simclr_smoke.yaml"))
    downstream = load_config(Path("configs/downstream/smoke.yaml"))
    full_pretrain = load_config(Path(matrix["pretrain_configs"]["simclr"]))
    source_config = Path(matrix["downstream_templates"]["ssl_linear"])
    source = load_config(source_config)
    pretrain.update(
        development_manifest=full_pretrain["development_manifest"],
        augmentation=full_pretrain["augmentation"],
        seed=full_pretrain["seed"],
        resume=True,
    )
    downstream.update(
        seed=source["seed"],
        encoder=pretrain["encoder"],
        image_size=pretrain["image_size"],
        resume=True,
    )
    inputs = {
        path: file_hash(path)
        for path in {
            full_pretrain["manifest"],
            full_pretrain["development_manifest"],
            source["train_manifest"],
            source["development_manifest"],
            source["sampled_patients"],
        }
    }
    fingerprint = config_signature(
        {
            "pretrain": pretrain,
            "downstream": downstream,
            "inputs": inputs,
            "pretrain_implementation": implementation_hash(Path("scripts/train_pretrain.py")),
            "downstream_implementation": implementation_hash(Path("scripts/train_downstream.py")),
        }
    )[:16]
    root = Path(pretrain["output_dir"]).parent / "runs" / fingerprint
    pretrain["output_dir"] = str(root / "pretrain")
    downstream.update(
        output_dir=str(root / "downstream"), ssl_checkpoint=str(root / "pretrain" / "best.pt")
    )
    return pretrain, downstream, root, full_pretrain["manifest"]


def notebook_smoke_suite(matrix):
    """Bounded checks of every SSL objective and all downstream initialization paths."""
    plan = build_plan(matrix)
    pretrain = plan["pretrain"][0]
    source = next(
        c
        for c in plan["downstream"]
        if c.get("ssl_method") == "simclr"
        and c["mode"] == "ssl_linear"
        and c["label_fraction"] == matrix.get("smoke_budget", 0.05)
    )
    inputs = {
        path: file_hash(path)
        for path in {
            pretrain["manifest"],
            pretrain["development_manifest"],
            source["train_manifest"],
            source["development_manifest"],
            source["sampled_patients"],
            pretrain["cache_source_manifest"],
        }
    }
    fingerprint = config_signature(
        {
            "matrix": matrix,
            "plan": plan,
            "inputs": inputs,
            "pretrain_implementation": implementation_hash(Path("scripts/train_pretrain.py")),
            "downstream_implementation": implementation_hash(Path("scripts/train_downstream.py")),
        }
    )[:16]
    root = Path(matrix["pretrain_output_root"]).parent / "smoke" / fingerprint
    data_root = Path(matrix["splits_dir"]).parent / "smoke" / fingerprint
    suite = {}
    for config in plan["pretrain"]:
        name = f"pretrain-{config['method']}"
        suite[name] = {
            **config,
            "output_dir": str(root / name),
            "epochs": 2,
            "batch_size": 64,
            "max_steps_per_epoch": 4,
            "manifest": str(data_root / "pretrain.csv"),
            "cache_path": str(data_root / "pretrain_gray.npy"),
            "resume": True,
        }
        suite[name].pop("cache_source_manifest", None)
    for name, method, mode in [
        ("linear", "simclr", "ssl_linear"),
        ("scratch", None, "supervised"),
        ("moco", "moco", "ssl_finetune"),
        ("imagenet", None, "supervised"),
    ]:
        config = next(
            c
            for c in plan["downstream"]
            if c["mode"] == mode
            and c.get("ssl_method") == method
            and c["imagenet_pretrained"] == (name == "imagenet")
        )
        config = {
            **config,
            "output_dir": str(root / f"downstream-{name}"),
            "epochs": 2,
            "batch_size": 64 if mode != "ssl_linear" else 512,
            "early_stopping_patience": 2,
            "label_fraction": 1.0,
            "resume": True,
            "train_manifest": str(data_root / "downstream_train.csv"),
            "development_manifest": str(data_root / "development.csv"),
            "sampled_patients": str(data_root / "cohort.json"),
            "training_cache_manifest": str(data_root / "all_training.csv"),
            "downstream_cache_path": str(data_root / "downstream_gray.npy"),
        }
        if method:
            config["ssl_checkpoint"] = str(root / f"pretrain-{method}" / "last.pt")
        suite[f"downstream-{name}"] = config
    return suite, root, pretrain, source


def run_notebook_smoke(matrix):
    suite, root, pretrain_source, downstream_source = notebook_smoke_suite(matrix)
    for name, config in suite.items():
        path = root / f"{name}.json"
        if path.exists() and load_config(path) != config:
            raise ValueError(f"Smoke configuration artifact changed: {path}")
        save_json(path, config)
    source_path = root / "source-downstream.json"
    save_json(source_path, downstream_source)
    data_root = Path(suite["downstream-linear"]["train_manifest"]).parent
    for command in [
        [sys.executable, "-m", "pytest", "-q"],
        [
            sys.executable,
            "scripts/prepare_smoke_manifest.py",
            "--source",
            pretrain_source["manifest"],
            "--output",
            suite["pretrain-simclr"]["manifest"],
        ],
        [
            sys.executable,
            "scripts/prepare_downstream_smoke.py",
            "--config",
            str(source_path),
            "--output-dir",
            str(data_root),
        ],
    ]:
        subprocess.run(command, check=True)
    for name, config in suite.items():
        script = "train_pretrain.py" if name.startswith("pretrain-") else "train_downstream.py"
        metadata = Path(config["output_dir"]) / "run_metadata.json"
        if metadata.exists() and load_config(metadata).get("status") == "complete":
            previous = load_config(metadata)["config"]
            if not all(previous.get(k) == v for k, v in config.items()):
                raise ValueError(f"Completed smoke configuration changed: {metadata}")
            for key, digest in previous.get("provenance", {}).items():
                if key in previous and file_hash(previous[key]) != digest:
                    raise ValueError(f"Completed smoke inputs changed: {metadata}")
            if not all(
                (Path(config["output_dir"]) / artifact).is_file()
                for artifact in ("best.pt", "last.pt", "metrics.json")
            ):
                raise ValueError(f"Completed smoke artifacts missing: {metadata}")
            print(f"Reusing completed smoke: {name}", flush=True)
            continue
        subprocess.run(
            [sys.executable, f"scripts/{script}", "--config", str(root / f"{name}.json")],
            check=True,
        )
    print(
        f"Inspect {suite['pretrain-simclr']['output_dir']}/augmentation_pairs.png before full pretraining."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/notebook_reference.yaml")
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
        "--partition",
        choices=("development", "final_validation"),
        default="final_validation",
        help="Partition for aggregate stage",
    )
    parser.add_argument(
        "--augmentation-reviewed",
        action="store_true",
        help="Confirm anatomical review of the completed smoke augmentation preview",
    )
    args = parser.parse_args()
    matrix = load_config(args.config)
    if args.stage == "prepare":
        subprocess.run(
            [sys.executable, "scripts/prepare_data.py", "--config", matrix["data_config"]],
            check=True,
        )
    elif args.stage == "smoke":
        if matrix.get("notebook_smoke"):
            run_notebook_smoke(matrix)
            return
        pretrain, downstream, smoke_root, source_manifest = smoke_settings(matrix)
        for name, config in (("pretrain", pretrain), ("downstream", downstream)):
            path = smoke_root / f"{name}.json"
            if path.exists() and load_config(path) != config:
                raise ValueError(f"Smoke configuration artifact changed: {path}")
            if not path.exists():
                save_json(path, config)
        for command in (
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "scripts/prepare_smoke_manifest.py", "--source", source_manifest],
            [
                sys.executable,
                "scripts/train_pretrain.py",
                "--config",
                str(smoke_root / "pretrain.json"),
            ],
            [
                sys.executable,
                "scripts/prepare_downstream_smoke.py",
                "--config",
                matrix["downstream_templates"]["ssl_linear"],
            ],
            [
                sys.executable,
                "scripts/train_downstream.py",
                "--config",
                str(smoke_root / "downstream.json"),
            ],
        ):
            subprocess.run(command, check=True)
        print(f"Inspect {pretrain['output_dir']}/augmentation_pairs.png before full pretraining.")
    elif args.stage == "aggregate":
        command = [sys.executable, "scripts/aggregate_results.py", "--partition", args.partition]
        if matrix.get("notebook_smoke"):
            command += [
                "--root",
                str(Path(matrix["pretrain_output_root"]).parent),
                "--output-dir",
                str(Path(matrix["pretrain_output_root"]).parent / "reports"),
                "--plots",
            ]
        subprocess.run(
            command,
            check=True,
        )
    if args.stage in {"prepare", "smoke", "aggregate"}:
        return
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
            if not path.exists():
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
    if args.stage in {"pretrain", "downstream", "evaluate"}:
        if args.stage == "pretrain":
            smoke, downstream_smoke, _, _ = smoke_settings(matrix)
            settings = (
                list(notebook_smoke_suite(matrix)[0].values())
                if matrix.get("notebook_smoke")
                else [smoke, downstream_smoke]
            )
            for setting in settings:
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
                    *(
                        ["last.pt"]
                        if args.stage == "pretrain"
                        and matrix.get("transfer_checkpoint") == "last.pt"
                        else []
                    ),
                ):
                    if not (Path(config["output_dir"]) / name).exists():
                        raise ValueError(f"Completed run is missing {name}: {metadata}")
                print(f"Reusing completed run: {config['output_dir']}", flush=True)
                continue
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
