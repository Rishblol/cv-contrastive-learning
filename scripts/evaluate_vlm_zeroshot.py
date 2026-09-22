"""Run a versioned, label-free VLM prompt baseline on a prepared manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from chexpert_ssl.data import TARGETS, target_columns
from chexpert_ssl.metrics import multilabel_metrics
from chexpert_ssl.utils import device_from_config, load_config, save_json, save_run_metadata
from chexpert_ssl.vlm import HuggingFaceVLM, aggregate_prompt_embeddings, zero_shot_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = device_from_config(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, config)
    with Path(config["prompt_set"]).open("r", encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle)["prompts"]
    model = HuggingFaceVLM(config["checkpoint"], config.get("revision")).to(device).eval()
    concepts = aggregate_prompt_embeddings(model, prompts, device)
    frame = pd.read_csv(config["manifest"])
    all_scores: list[np.ndarray] = []
    batch_size = int(config.get("batch_size", 32))
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            paths = frame.iloc[start : start + batch_size]["image_path"].tolist()
            images = [Image.open(path).convert("RGB") for path in paths]
            batch = model.processor(images=images, return_tensors="pt")
            for image in images:
                image.close()
            embeddings = model.image_features(batch["pixel_values"].to(device))
            all_scores.append(zero_shot_scores(embeddings, concepts))
    scores = np.concatenate(all_scores)
    targets = frame[target_columns()].to_numpy(dtype=np.float32)
    save_json(output_dir / "metrics.json", {**multilabel_metrics(targets, scores), "calibrated": False, "prompt_set": config["prompt_set"]})
    output = frame[["image_path", "patient_id", "study_id", "Frontal/Lateral", *target_columns()]].copy()
    for index, label in enumerate(TARGETS):
        output[f"probability_{label}"] = scores[:, index]
    output.to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
