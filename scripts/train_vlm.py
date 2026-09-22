"""Train a frozen VLM feature probe or permitted VLM image-encoder fine-tune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from chexpert_ssl.data import TARGETS, sample_patient_budget, target_columns
from chexpert_ssl.metrics import multilabel_metrics, select_thresholds
from chexpert_ssl.utils import device_from_config, load_config, save_json, save_run_metadata, seed_everything
from chexpert_ssl.vlm import HuggingFaceVLM, VLMClassifier


class Records(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
    def __len__(self) -> int:
        return len(self.frame)
    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
        return image, torch.tensor(row[target_columns()].to_numpy(dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    mode = config["mode"]
    if mode not in {"vlm_linear", "vlm_finetune"}:
        raise ValueError("mode must be vlm_linear or vlm_finetune")
    seed_everything(int(config["seed"]))
    device = device_from_config(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, config)
    train = pd.read_csv(config["train_manifest"])
    if config.get("sampled_patients"):
        patient_ids = set(str(item) for item in json.loads(Path(config["sampled_patients"]).read_text())["patient_ids"])
        train = train.loc[train.patient_id.astype(str).isin(patient_ids)].copy()
    else:
        train = sample_patient_budget(train, float(config["label_fraction"]), int(config["seed"]))
    dev = pd.read_csv(config["development_manifest"])
    encoder = HuggingFaceVLM(config["checkpoint"], config.get("revision"))
    model = VLMClassifier(encoder, len(TARGETS), freeze_encoder=mode == "vlm_linear").to(device)

    def collate(rows):
        images, targets = zip(*rows)
        pixels = encoder.processor(images=list(images), return_tensors="pt")["pixel_values"]
        return pixels, torch.stack(list(targets))
    train_loader = DataLoader(Records(train), batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(Records(dev), batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate)
    values = train[target_columns()].to_numpy(dtype=np.float32)
    pos_weight = torch.tensor((len(values) - values.sum(0)) / np.clip(values.sum(0), 1, None), device=device)
    if mode == "vlm_finetune":
        optimizer = AdamW(
            [
                {"params": model.encoder.parameters(), "lr": float(config.get("encoder_learning_rate", config["learning_rate"]))},
                {"params": model.classifier.parameters(), "lr": float(config["learning_rate"])},
            ],
            weight_decay=float(config["weight_decay"]),
        )
    else:
        optimizer = AdamW(model.classifier.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_epoch, history = -np.inf, 0, []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for pixels, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(pixels.to(device)), targets.to(device))
            loss.backward(); optimizer.step()
        model.eval(); scores, truths = [], []
        with torch.inference_mode():
            for pixels, targets in dev_loader:
                scores.append(torch.sigmoid(model(pixels.to(device))).cpu().numpy()); truths.append(targets.numpy())
        truth, score = np.concatenate(truths), np.concatenate(scores)
        metrics = multilabel_metrics(truth, score); metrics["epoch"] = epoch; history.append(metrics)
        if np.isfinite(metrics["macro_auroc"]) and metrics["macro_auroc"] > best:
            best, best_epoch = metrics["macro_auroc"], epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "best.pt")
            save_json(output_dir / "thresholds.json", select_thresholds(truth, score))
        if epoch - best_epoch >= int(config.get("early_stopping_patience", 10)):
            break
    save_json(output_dir / "metrics.json", {"mode": mode, "label_fraction": config["label_fraction"], "best_dev_auroc": best, "best_epoch": best_epoch, "history": history, "positive_weights": pos_weight.cpu().tolist()})
    train[["patient_id"]].drop_duplicates().to_csv(output_dir / "sampled_patients.csv", index=False)


if __name__ == "__main__":
    main()
