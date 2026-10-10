"""Train matched supervised, SSL frozen-probe, and SSL fine-tuning experiments."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from chexpert_ssl.data import (
    TARGETS,
    CheXpertDataset,
    assert_patient_disjoint,
    read_manifest,
    reject_final_partition,
    select_patient_cohort,
    supervised_transform,
    target_columns,
)
from chexpert_ssl.metrics import multilabel_metrics, select_thresholds
from chexpert_ssl.models import MultiLabelClassifier
from chexpert_ssl.utils import (
    atomic_torch_save,
    capture_rng_state,
    config_signature,
    device_from_config,
    file_hash,
    finish_run,
    load_config,
    managed_run,
    restore_rng_state,
    save_json,
    save_run_metadata,
    seed_everything,
    validate_training_config,
)


def positive_weights(frame: pd.DataFrame) -> torch.Tensor:
    targets = frame[target_columns()].to_numpy(dtype=np.float32)
    positives = targets.sum(axis=0)
    return torch.tensor(
        (len(targets) - positives) / np.clip(positives, 1, None), dtype=torch.float32
    )


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths, scores = [], []
    for inputs, targets in loader:
        truths.append(targets.numpy())
        scores.append(torch.sigmoid(model(inputs.to(device))).float().cpu().numpy())
    return np.concatenate(truths), np.concatenate(scores)


def extract_features(
    model: MultiLabelClassifier,
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    multiprocessing_context: str = "spawn",
) -> TensorDataset:
    loader = DataLoader(
        CheXpertDataset(frame, supervised_transform(image_size, False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        multiprocessing_context=multiprocessing_context if workers else None,
    )
    model.encoder.eval()
    features, labels = [], []
    for images, targets in loader:
        with torch.no_grad():
            features.append(model.encoder(images.to(device)).float().cpu())
        labels.append(targets)
    return TensorDataset(torch.cat(features), torch.cat(labels))


@managed_run
def train(config: dict) -> None:
    validate_training_config(config, "downstream")
    seed_everything(int(config["seed"]))
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    device = device_from_config(config.get("device", "auto"))
    amp = device.type == "cuda" and config["precision"] == "auto"
    config = dict(config)
    config["effective_precision"] = "float16" if amp else "float32"
    train_frame = read_manifest(config["train_manifest"])
    dev_frame = read_manifest(config["development_manifest"])
    reject_final_partition(config["train_manifest"], train_frame)
    reject_final_partition(config["development_manifest"], dev_frame)
    assert_patient_disjoint(train_frame, dev_frame)
    train_frame = select_patient_cohort(
        train_frame,
        config["sampled_patients"],
        float(config["label_fraction"]),
        int(config["seed"]),
    )
    mode = config["mode"]
    is_ssl = mode != "supervised"
    linear = mode in {"ssl_linear", "simclr_linear"}
    actual_method = config.get("ssl_method") if is_ssl else None
    provenance = {
        key: file_hash(config[key])
        for key in ("train_manifest", "development_manifest", "sampled_patients")
    }
    if is_ssl:
        provenance["ssl_checkpoint"] = file_hash(config["ssl_checkpoint"])
    weights = positive_weights(train_frame)
    config["provenance"] = provenance
    config["positive_weights"] = weights.tolist()
    signature = config_signature(config)
    output_dir = Path(config["output_dir"])
    resume_path = output_dir / "last.pt"
    resumed = None
    if resume_path.exists():
        if not config["resume"]:
            raise ValueError("Output directory contains a checkpoint; choose a new output_dir")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resumed.get("training_signature") != signature:
            raise ValueError(
                "Resume configuration, data cohort, or initialization checkpoint changed"
            )
    elif any((output_dir / name).exists() for name in ("best.pt", "metrics.json")):
        raise ValueError(
            "Existing run artifacts without a resumable checkpoint; choose a new output_dir"
        )
    if (output_dir / "selection_lock.json").exists():
        raise ValueError("A selected run is immutable; use a new output_dir for further training")

    model = MultiLabelClassifier(
        config["encoder"],
        len(TARGETS),
        freeze_encoder=linear,
        pretrained=config["imagenet_pretrained"] and resumed is None,
        weights_name=config.get("imagenet_weights"),
    )
    if is_ssl:
        source = torch.load(config["ssl_checkpoint"], map_location="cpu", weights_only=False)
        if source.get("method") != actual_method:
            raise ValueError("SSL method does not match initialization checkpoint")
        source_config = source.get("config", {})
        for key in ("encoder", "image_size"):
            if source_config.get(key) != config[key]:
                raise ValueError(f"SSL checkpoint {key} does not match downstream comparison")
        model.load_pretrained_encoder(source)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not linear and config.get("encoder_learning_rate") is not None:
        parameters = [
            {"params": model.encoder.parameters(), "lr": float(config["encoder_learning_rate"])},
            {"params": model.classifier.parameters()},
        ]
    optimizer = AdamW(
        parameters, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    start_epoch, best_score, best_epoch, history = 1, -np.inf, 0, []
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        start_epoch = int(resumed["epoch"]) + 1
        best_score, best_epoch, history = (
            resumed["best_score"],
            resumed["best_epoch"],
            resumed["history"],
        )

    size, batch, workers = (
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
    )
    if linear:
        print("Extracting frozen training/development features once", flush=True)
        train_data = extract_features(
            model, train_frame, size, batch, workers, device, config["multiprocessing_context"]
        )
        dev_data = extract_features(
            model, dev_frame, size, batch, workers, device, config["multiprocessing_context"]
        )
        train_model = model.classifier
        loader_workers = 0
    else:
        train_data = CheXpertDataset(
            train_frame, supervised_transform(size, True, config["supervised_augmentation"])
        )
        dev_data = CheXpertDataset(dev_frame, supervised_transform(size, False))
        train_model = model
        loader_workers = workers
    shuffle_generator = torch.Generator()
    worker_generator = torch.Generator()
    train_loader = DataLoader(
        train_data,
        batch_size=batch,
        sampler=RandomSampler(train_data, generator=shuffle_generator),
        generator=worker_generator,
        num_workers=loader_workers,
        multiprocessing_context=config["multiprocessing_context"] if loader_workers else None,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch,
        shuffle=False,
        num_workers=loader_workers,
        multiprocessing_context=config["multiprocessing_context"] if loader_workers else None,
    )
    if resumed is not None:
        restore_rng_state(resumed["rng_state"])
    save_run_metadata(output_dir, config)
    train_frame[["patient_id"]].drop_duplicates().to_csv(
        output_dir / "sampled_patients.csv", index=False
    )
    total_started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        shuffle_generator.manual_seed(int(config["seed"]) + epoch)
        worker_generator.manual_seed(int(config["seed"]) + epoch)
        if history and epoch - 1 - best_epoch >= int(config["early_stopping_patience"]):
            break
        started = time.perf_counter()
        train_model.train()
        if linear:
            model.encoder.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        loss_sum, records = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and not linear):
                loss = criterion(train_model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(inputs)
            records += len(inputs)
        truths, scores = predict(train_model, dev_loader, device)
        metrics = multilabel_metrics(truths, scores)
        metrics.update(
            epoch=epoch,
            train_loss=loss_sum / records,
            seconds=time.perf_counter() - started,
            peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0,
        )
        history.append(metrics)
        score = metrics["macro_auroc"]
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.4f} dev_auroc={score:.4f}",
            flush=True,
        )
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch = score, epoch
            thresholds = select_thresholds(truths, scores)
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "training_signature": signature,
                    "thresholds": thresholds,
                },
                output_dir / "best.pt",
            )
            save_json(output_dir / "thresholds.json", thresholds)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "training_signature": signature,
                "rng_state": capture_rng_state(),
            },
            resume_path,
        )
    if not math.isfinite(best_score):
        raise ValueError("Development AUROC is undefined; no selectable checkpoint was produced")
    best_metrics = next(row for row in history if row["epoch"] == best_epoch)
    save_json(
        output_dir / "metrics.json",
        {
            **best_metrics,
            "partition": "development",
            "mode": mode,
            "ssl_method": actual_method,
            "label_fraction": config["label_fraction"],
            "seed": config["seed"],
            "history": history,
            "best_dev_auroc": best_score,
            "best_epoch": best_epoch,
            "positive_weights": weights.tolist(),
            "training_signature": signature,
        },
    )
    finish_run(
        output_dir,
        epochs_completed=history[-1]["epoch"],
        invocation_seconds=time.perf_counter() - total_started,
        training_seconds=sum(row["seconds"] for row in history),
        checkpoint_path=str(output_dir / "best.pt"),
        checkpoint_bytes=(output_dir / "best.pt").stat().st_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
