"""Evaluate a development-locked image-only checkpoint on official validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chexpert_ssl.data import (
    TARGETS,
    CheXpertDataset,
    assert_patient_disjoint,
    read_manifest,
    supervised_transform,
    target_columns,
)
from chexpert_ssl.metrics import (
    multilabel_metrics,
    patient_bootstrap_auroc,
    patient_bootstrap_macro_auroc,
    threshold_metrics,
)
from chexpert_ssl.models import MultiLabelClassifier
from chexpert_ssl.selection import verify_selection
from chexpert_ssl.utils import (
    device_from_config,
    file_hash,
    finish_run,
    load_config,
    managed_run,
    save_json,
    save_run_metadata,
)


@managed_run
def evaluate(config: dict) -> None:
    config = dict(config)
    required = {
        "checkpoint",
        "selection_lock",
        "thresholds",
        "manifest",
        "output_dir",
        "batch_size",
        "seed",
        "bootstrap_samples",
    }
    if required.difference(config):
        raise ValueError(f"Missing evaluation keys: {sorted(required.difference(config))}")
    if int(config["batch_size"]) < 1 or int(config["bootstrap_samples"]) < 1:
        raise ValueError("Evaluation batch size and bootstrap sample count must be positive")
    output_dir = Path(config["output_dir"])
    if (output_dir / "metrics.json").exists():
        raise ValueError("Evaluation results already exist; reuse them instead of reevaluating")
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    training, lock = verify_selection(config, checkpoint)
    config["training_config"] = training
    config["source_run_id"] = lock["source_run_id"]
    for key in ("encoder", "image_size", "target_policy", "supervised_augmentation"):
        config[key] = training[key]
    train = read_manifest(training["train_manifest"])
    dev = read_manifest(training["development_manifest"])
    for key in ("train_manifest", "development_manifest"):
        if file_hash(training[key]) != training["provenance"][key]:
            raise ValueError("Training/development manifest changed since selection")
    frame = read_manifest(config["manifest"])
    assert_patient_disjoint(train, dev, frame)
    config["provenance"] = {
        key: file_hash(config[key])
        for key in ("checkpoint", "selection_lock", "thresholds", "manifest")
    }
    device = device_from_config(config.get("device", "auto"))
    # All weights come from the selected checkpoint; evaluation never downloads initialization weights.
    model = MultiLabelClassifier(
        training["encoder"],
        len(TARGETS),
        pretrained=False,
        standardize_features=training.get("standardize_features", False)
        and training["mode"] in {"ssl_linear", "simclr_linear"},
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    loader = DataLoader(
        CheXpertDataset(
            frame,
            supervised_transform(
                training["image_size"], False, training.get("supervised_augmentation")
            ),
        ),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        multiprocessing_context=training["multiprocessing_context"]
        if int(config.get("num_workers", 0))
        else None,
    )
    save_run_metadata(output_dir, config)
    probabilities = []
    with torch.inference_mode():
        for images, _ in loader:
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda"
                and training.get("cached_downstream")
                and training["precision"] == "auto",
            ):
                logits = model(images.to(device))
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    scores = np.concatenate(probabilities)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite evaluation probabilities")
    targets = frame[target_columns()].to_numpy(dtype=np.float32)
    thresholds = load_config(Path(config["thresholds"]))
    metrics = multilabel_metrics(targets, scores)
    metrics.update(
        threshold_metrics(
            targets, scores, thresholds, notebook=training.get("threshold_method") == "youden"
        )
    )
    if config.get("per_label_bootstrap"):
        intervals = patient_bootstrap_auroc(
            targets,
            scores,
            frame.patient_id.to_numpy(),
            int(config["seed"]),
            int(config["bootstrap_samples"]),
        )
        metrics["macro_auroc_ci95"] = intervals["macro_auroc"]
        metrics["per_label_auroc_ci95"] = intervals["per_label"]
    else:
        metrics["macro_auroc_ci95"] = patient_bootstrap_macro_auroc(
            targets,
            scores,
            frame.patient_id.to_numpy(),
            int(config["seed"]),
            int(config["bootstrap_samples"]),
        )
    metrics.update(
        partition="final_validation",
        source_run_id=lock["source_run_id"],
        mode=training["mode"],
        ssl_method=training.get("ssl_method"),
        label_fraction=training["label_fraction"],
        seed=training["seed"],
        training_signature=checkpoint["training_signature"],
    )
    columns = [
        "image_path",
        "patient_id",
        "study_id",
        "Frontal/Lateral",
        "AP/PA",
        "Age",
        "Sex",
        *target_columns(),
    ]
    predictions = frame[[column for column in columns if column in frame]].copy()
    for index, label in enumerate(TARGETS):
        predictions[f"probability_{label}"] = scores[:, index]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_json(output_dir / "metrics.json", metrics)
    finish_run(
        output_dir, evaluated_images=len(frame), evaluated_patients=frame.patient_id.nunique()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    evaluate(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
