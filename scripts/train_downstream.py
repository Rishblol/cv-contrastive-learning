"""Train supervised, SimCLR-linear-probe, or SimCLR-fine-tuning classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chexpert_ssl.data import CheXpertDataset, TARGETS, sample_patient_budget, supervised_transform, target_columns
from chexpert_ssl.metrics import multilabel_metrics, select_thresholds
from chexpert_ssl.models import MultiLabelClassifier
from chexpert_ssl.utils import device_from_config, load_config, save_json, save_run_metadata, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_probabilities = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        all_targets.append(targets.numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


def positive_weights(frame: pd.DataFrame) -> torch.Tensor:
    targets = frame[target_columns()].to_numpy(dtype=np.float32)
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    return torch.tensor(negatives / np.clip(positives, 1, None), dtype=torch.float32)


def extract_features(model: MultiLabelClassifier, frame: pd.DataFrame, image_size: int,
                     batch_size: int, workers: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        CheXpertDataset(frame, supervised_transform(image_size, train=False)),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
    )
    model.encoder.eval()
    features, labels = [], []
    for images, targets in loader:
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            features.append(model.encoder(images.to(device, non_blocking=True)).float().cpu())
        labels.append(targets)
    with torch.inference_mode(False):
        train_features = torch.cat(features).clone()
    return train_features, torch.cat(labels)


def main() -> None:
    config = load_config(parse_args().config)
    seed_everything(int(config["seed"]))
    device = device_from_config(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, config)

    train_frame = pd.read_csv(config["train_manifest"])
    dev_frame = pd.read_csv(config["development_manifest"])
    if config.get("sampled_patients"):
        with Path(config["sampled_patients"]).open("r", encoding="utf-8") as handle:
            patient_ids = set(str(value) for value in json.load(handle)["patient_ids"])
        train_frame = train_frame.loc[train_frame["patient_id"].astype(str).isin(patient_ids)].copy()
    else:
        train_frame = sample_patient_budget(train_frame, float(config["label_fraction"]), int(config["seed"]))
    image_size = int(config["image_size"])
    train_loader = DataLoader(
        CheXpertDataset(train_frame, supervised_transform(image_size, train=True)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        CheXpertDataset(dev_frame, supervised_transform(image_size, train=False)),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    mode = config["mode"]
    is_ssl = mode in {"simclr_linear", "simclr_finetune", "ssl_linear", "ssl_finetune"}
    linear_probe = mode in {"simclr_linear", "ssl_linear"}
    model = MultiLabelClassifier(
        config.get("encoder", "resnet50"),
        len(TARGETS),
        freeze_encoder=linear_probe,
        pretrained=bool(config.get("imagenet_pretrained", False)),
    )
    if is_ssl:
        checkpoint_path = config.get("ssl_checkpoint", config.get("simclr_checkpoint"))
        if not checkpoint_path:
            raise ValueError("SSL downstream runs require ssl_checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        requested_method = config.get("ssl_method")
        actual_method = checkpoint.get("method", "simclr")
        if requested_method and requested_method != actual_method:
            raise ValueError(f"Configured SSL method {requested_method} does not match checkpoint {actual_method}")
        model.load_pretrained_encoder(checkpoint)
    elif mode != "supervised":
        raise ValueError("mode must be supervised, ssl_linear, or ssl_finetune")
    model.to(device)

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights(train_frame).to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = float("-inf")
    best_epoch = 0
    patience = int(config.get("early_stopping_patience", 10))
    history: list[dict[str, float]] = []
    cached_train = cached_dev = None
    if linear_probe:
        print("Extracting frozen train/development features once for the linear probe...")
        cached_train = extract_features(model, train_frame, image_size, int(config["batch_size"]),
                                        int(config.get("num_workers", 4)), device)
        cached_dev = extract_features(model, dev_frame, image_size, int(config["batch_size"]),
                                      int(config.get("num_workers", 4)), device)
    start_epoch = 1
    resume_path = output_dir / "last.pt"
    if bool(config.get("resume", True)) and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        history = checkpoint["history"]

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train(not linear_probe)
        if linear_probe:
            model.encoder.eval()
        loss_sum = 0.0
        if linear_probe:
            features, targets = cached_train
            permutation = torch.randperm(len(features))
            for indices in permutation.split(int(config["batch_size"])):
                batch_features = features[indices].to(device, non_blocking=True)
                batch_targets = targets[indices].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model.classifier(batch_features), batch_targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            dev_targets = cached_dev[1].numpy()
            model.classifier.eval()
            with torch.inference_mode():
                dev_probabilities = torch.sigmoid(
                    model.classifier(cached_dev[0].to(device)).float()
                ).cpu().numpy()
        else:
            for images, targets in train_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()
            dev_targets, dev_probabilities = predict(model, dev_loader, device)
        metrics = multilabel_metrics(dev_targets, dev_probabilities)
        metrics.update({"epoch": epoch, "train_loss": loss_sum / max(1, len(train_loader))})
        history.append(metrics)
        score = metrics["macro_auroc"]
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} dev_auroc={score:.4f}")
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "best.pt")
            save_json(output_dir / "thresholds.json", select_thresholds(dev_targets, dev_probabilities))
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
            },
            resume_path,
        )
        if epoch - best_epoch >= patience:
            break

    save_json(
        output_dir / "metrics.json",
        {"mode": mode, "ssl_method": actual_method if is_ssl else None,
         "label_fraction": config["label_fraction"], "history": history,
         "best_dev_auroc": best_score, "best_epoch": best_epoch,
         "positive_weights": positive_weights(train_frame).tolist()},
    )
    train_frame[["patient_id"]].drop_duplicates().to_csv(output_dir / "sampled_patients.csv", index=False)


if __name__ == "__main__":
    main()
