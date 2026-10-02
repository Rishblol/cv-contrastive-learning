"""Evaluate a locked downstream checkpoint once on a supplied manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chexpert_ssl.data import CheXpertDataset, TARGETS, supervised_transform, target_columns
from chexpert_ssl.metrics import multilabel_metrics, patient_bootstrap_macro_auroc, threshold_metrics
from chexpert_ssl.models import MultiLabelClassifier
from chexpert_ssl.utils import device_from_config, load_config, save_json, save_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = device_from_config(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, config)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model = MultiLabelClassifier(
        config.get("encoder", "resnet50"),
        len(TARGETS),
        pretrained=bool(config.get("imagenet_pretrained", False)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    frame = pd.read_csv(config["manifest"])
    loader = DataLoader(CheXpertDataset(frame, supervised_transform(int(config["image_size"]), False)), batch_size=int(config["batch_size"]), shuffle=False)
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in loader:
            probabilities.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    scores = np.concatenate(probabilities)
    targets = frame[target_columns()].to_numpy(dtype=np.float32)
    metrics = multilabel_metrics(targets, scores)
    thresholds = load_config(Path(config["thresholds"])) if config.get("thresholds") else {label: 0.5 for label in TARGETS}
    metrics.update(threshold_metrics(targets, scores, thresholds))
    metrics["macro_auroc_ci95"] = patient_bootstrap_macro_auroc(targets, scores, frame["patient_id"].to_numpy(), int(config.get("seed", 42)), int(config.get("bootstrap_samples", 1000)))
    save_json(output_dir / "metrics.json", metrics)
    predictions = frame[["image_path", "patient_id", "study_id", "Frontal/Lateral", *target_columns()]].copy()
    for index, label in enumerate(TARGETS):
        predictions[f"probability_{label}"] = scores[:, index]
    predictions.to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
