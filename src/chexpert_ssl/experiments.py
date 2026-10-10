"""Build a reproducible image-only experiment matrix from resolved YAML templates."""

from __future__ import annotations

from pathlib import Path

from .ssl_methods import METHODS
from .utils import load_config, validate_training_config


def build_plan(matrix: dict) -> dict[str, list[dict]]:
    if set(matrix["pretrain_configs"]) != set(METHODS):
        raise ValueError("The primary matrix must include SimCLR, MoCo, BYOL, NNCLR, and SwAV")
    if set(matrix["downstream_templates"]) != {"ssl_linear", "ssl_finetune", "scratch", "imagenet"}:
        raise ValueError("Matrix must include both SSL protocols and scratch/ImageNet references")
    budgets, seeds = matrix["budgets"], matrix["seeds"]
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(not 0 < float(b) <= 1 for b in budgets)
    ):
        raise ValueError("Budgets must be unique fractions in (0, 1]")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) for s in seeds):
        raise ValueError("Seeds must be unique integers")
    pretrain = []
    shared_pretrain = (
        "seed",
        "encoder",
        "image_size",
        "batch_size",
        "epochs",
        "manifest",
        "development_manifest",
        "augmentation",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "schedule",
        "precision",
        "target_policy",
        "cache_path",
    )
    for method in METHODS:
        config = load_config(Path(matrix["pretrain_configs"][method]))
        validate_training_config(config, "pretrain")
        if config["method"] != method:
            raise ValueError("Pretraining config method does not match matrix arm")
        if pretrain and any(config.get(key) != pretrain[0].get(key) for key in shared_pretrain):
            raise ValueError(
                "All five pretraining methods must share the primary comparison settings"
            )
        pretrain.append(config)
    templates = {
        key: load_config(Path(path)) for key, path in matrix["downstream_templates"].items()
    }
    common = (
        "encoder",
        "image_size",
        "train_manifest",
        "development_manifest",
        "target_policy",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "supervised_augmentation",
        "precision",
    )
    reference = templates["scratch"]
    for config in templates.values():
        if any(config.get(key) != reference.get(key) for key in common):
            raise ValueError("Downstream arms must share data and evaluation settings")
        if any(
            config[key] != pretrain[0][key] for key in ("encoder", "image_size", "target_policy")
        ):
            raise ValueError(
                "Downstream architecture, resolution, and targets must match pretraining"
            )
    downstream, evaluation = [], []
    for budget in sorted(budgets):
        fraction = float(budget)
        token = f"{fraction:g}".replace(".", "p")
        for seed in seeds:
            cohort = str(Path(matrix["splits_dir"]) / f"budget_{token}_seed_{seed}.json")
            arms = [
                (method, protocol)
                for method in METHODS
                for protocol in ("ssl_linear", "ssl_finetune")
            ]
            arms += [("scratch", "scratch"), ("imagenet", "imagenet")]
            for method, protocol in arms:
                config = dict(templates[protocol])
                config.update(seed=seed, label_fraction=fraction, sampled_patients=cohort)
                if method in METHODS:
                    source = next(c for c in pretrain if c["method"] == method)
                    config.update(
                        mode=protocol,
                        ssl_method=method,
                        ssl_checkpoint=str(Path(source["output_dir"]) / "best.pt"),
                    )
                else:
                    config.pop("ssl_method", None)
                    config.pop("ssl_checkpoint", None)
                    config.update(mode="supervised", imagenet_pretrained=method == "imagenet")
                run_id = f"{method}-{protocol}-{config['encoder']}-budget{token}-seed{seed}"
                config["output_dir"] = str(Path(matrix["downstream_output_root"]) / run_id)
                validate_training_config(config, "downstream")
                downstream.append(config)
                evaluation.append(
                    {
                        "device": config["device"],
                        "seed": seed,
                        "checkpoint": str(Path(config["output_dir"]) / "best.pt"),
                        "thresholds": str(Path(config["output_dir"]) / "thresholds.json"),
                        "selection_lock": str(Path(config["output_dir"]) / "selection_lock.json"),
                        "manifest": matrix["final_manifest"],
                        "batch_size": config["batch_size"],
                        "num_workers": config["num_workers"],
                        "bootstrap_samples": matrix["bootstrap_samples"],
                        "output_dir": str(Path(matrix["evaluation_output_root"]) / run_id),
                    }
                )
    directories = [c["output_dir"] for c in (*pretrain, *downstream, *evaluation)]
    if len(set(directories)) != len(directories):
        raise ValueError("Experiment output directories must be unique")
    return {"pretrain": pretrain, "downstream": downstream, "evaluate": evaluation}
