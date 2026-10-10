"""Train one SSL method with a reusable image cache and device-side augmentation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import save_image

from chexpert_ssl.data import assert_patient_disjoint, read_manifest, reject_final_partition
from chexpert_ssl.gpu_augment import augment_grayscale_batch
from chexpert_ssl.image_cache import ensure_grayscale_cache
from chexpert_ssl.ssl_methods import METHODS, build_ssl_method, encoder_state_dict
from chexpert_ssl.utils import (
    artifact_lock,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def training_signature(config: dict, method: str, cache_metadata: dict) -> str:
    return config_signature({**config, "method": method, "cache": cache_metadata})


def save_augmentation_preview(
    dataset,
    image_size: int,
    augmentation: dict,
    device: torch.device,
    channels_last: bool,
    output_path: Path,
    seed: int,
) -> None:
    count = min(8, len(dataset))
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False))
    grayscale = torch.from_numpy(np.stack([dataset[int(index)] for index in indices])).to(device)
    original = grayscale.float().div(255).sub(0.5).div(0.5).unsqueeze(1).expand(-1, 3, -1, -1)
    first = augment_grayscale_batch(grayscale, image_size, augmentation, channels_last)
    second = augment_grayscale_batch(grayscale, image_size, augmentation, channels_last)
    preview = torch.cat((original, first, second)).cpu()
    save_image(preview, output_path, normalize=True, value_range=(-1, 1), nrow=count)


@managed_run
def train(config: dict) -> None:
    config = dict(config)
    validate_training_config(config, "pretrain")
    method_name = str(config.get("method", "simclr")).lower()
    if method_name not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")

    seed_everything(int(config["seed"]))
    device = device_from_config(config.get("device", "auto"))
    amp = device.type == "cuda" and config["precision"] == "auto"
    config["effective_precision"] = "float16" if amp else "float32"
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    torch.backends.cudnn.deterministic = bool(config.get("deterministic", False))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
        torch.backends.cudnn.allow_tf32 = bool(config.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))
        torch.set_float32_matmul_precision("high")

    output_dir = Path(config["output_dir"])
    if not (output_dir / "last.pt").exists() and any(
        (output_dir / name).exists() for name in ("best.pt", "metrics.json")
    ):
        raise ValueError("Existing run artifacts without a resumable checkpoint")
    if not config["resume"] and any(
        (output_dir / name).exists() for name in ("last.pt", "best.pt", "metrics.json")
    ):
        raise ValueError("Output directory already contains training artifacts")
    frame = read_manifest(config["manifest"], labeled=False)
    development = read_manifest(config["development_manifest"], labeled=False)
    reject_final_partition(config["manifest"], frame)
    reject_final_partition(config["development_manifest"], development)
    assert_patient_disjoint(frame, development)
    config["identifier_normalization"] = {
        "manifest_restored_rows": frame.attrs.get("patient_id_padding_restored_rows", 0),
        "development_restored_rows": development.attrs.get("patient_id_padding_restored_rows", 0),
    }
    config["provenance"] = {
        key: file_hash(config[key]) for key in ("manifest", "development_manifest")
    }
    if len(frame) < int(config["batch_size"]):
        raise ValueError("Pretraining manifest must contain at least one full batch")

    image_size = int(config["image_size"])
    cache_path = Path(
        config.get("cache_path", f"data/processed/cache/pretrain_gray_{image_size}.npy")
    )
    with artifact_lock(cache_path.with_suffix(".lock")):
        dataset, cache_built = ensure_grayscale_cache(
            frame,
            cache_path,
            image_size,
            workers=int(config.get("cache_workers", 16)),
            log_interval=int(config.get("cache_log_interval", 10_000)),
        )
        cache_metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    signature = training_signature(config, method_name, cache_metadata)

    if device.type == "cuda":
        print(
            f"device=cuda GPU={torch.cuda.get_device_name(device)} "
            f"VRAM_GB={torch.cuda.get_device_properties(device).total_memory / 2**30:.1f}",
            flush=True,
        )
    else:
        print("device=cpu; CUDA is unavailable", flush=True)
    print(
        f"method={method_name} images={len(dataset):,} image_size={image_size} "
        f"batch_size={config['batch_size']} cache={'built' if cache_built else 'reused'} "
        f"cache_path={cache_path}",
        flush=True,
    )

    augmentation = config.get("augmentation", {})
    channels_last = bool(config.get("channels_last", True)) and device.type == "cuda"

    workers = int(config.get("num_workers", 8))
    shuffle_generator = torch.Generator()
    worker_generator = torch.Generator()
    loader_options = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "sampler": RandomSampler(dataset, generator=shuffle_generator),
        "generator": worker_generator,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = int(config.get("prefetch_factor", 4))
        loader_options["multiprocessing_context"] = config["multiprocessing_context"]
    loader = DataLoader(**loader_options)

    model = build_ssl_method(method_name, config.get("encoder", "resnet18"), config.get("ssl", {}))
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    fused = bool(config.get("fused_optimizer", True)) and device.type == "cuda"
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=fused,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch = 1
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    resume_path = output_dir / "last.pt"
    if bool(config.get("resume", True)) and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("training_signature") != signature:
            raise ValueError(
                "Checkpoint settings do not match this optimized training configuration. "
                "Use a new output_dir or restore the exact settings used to create it."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        history = checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"])
        print(f"Resuming {method_name} at epoch {start_epoch}/{config['epochs']}", flush=True)

    save_run_metadata(output_dir, config)
    if not history:
        save_augmentation_preview(
            dataset,
            image_size,
            augmentation,
            device,
            channels_last,
            output_dir / "augmentation_pairs.png",
            int(config["seed"]),
        )
    print(
        f"mixed_precision={amp} channels_last={channels_last} "
        f"workers={workers} prefetch_factor={config.get('prefetch_factor', 4)} "
        f"fused_optimizer={fused}",
        flush=True,
    )

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        shuffle_generator.manual_seed(int(config["seed"]) + epoch)
        worker_generator.manual_seed(int(config["seed"]) + epoch)
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        log_every = max(1, len(loader) // 10)
        interval_started = time.perf_counter()
        last_logged_step = 0
        for batch_index, grayscale in enumerate(loader, start=1):
            grayscale = grayscale.to(device, non_blocking=True)
            first = augment_grayscale_batch(grayscale, image_size, augmentation, channels_last)
            second = augment_grayscale_batch(grayscale, image_size, augmentation, channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model(first, second)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite {method_name} loss at epoch {epoch}, step {batch_index}"
                )
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                model.after_optimizer_step((epoch - 1) / max(1, int(config["epochs"])))
            total_loss.add_(loss.detach().float())
            if batch_index % log_every == 0 or batch_index == len(loader):
                if device.type == "cuda":
                    allocated_gb = torch.cuda.memory_allocated(device) / 2**30
                else:
                    allocated_gb = 0.0
                reported_loss = float(loss.detach().item())
                interval_seconds = time.perf_counter() - interval_started
                interval_steps = batch_index - last_logged_step
                print(
                    f"method={method_name} epoch={epoch} step={batch_index}/{len(loader)} "
                    f"loss={reported_loss:.4f} steps_per_sec={interval_steps / interval_seconds:.2f} "
                    f"gpu_memory_gb={allocated_gb:.2f}",
                    flush=True,
                )
                last_logged_step = batch_index
                interval_started = time.perf_counter()

        scheduler.step()
        mean_loss = float((total_loss / max(1, len(loader))).item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / 2**30
        else:
            peak_memory_gb = 0.0
        epoch_seconds = time.perf_counter() - epoch_started
        images_per_second = len(loader) * int(config["batch_size"]) / max(epoch_seconds, 1e-9)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "seconds": epoch_seconds,
                "images_per_second": images_per_second,
                "peak_gpu_memory_gb": peak_memory_gb,
            }
        )
        print(
            f"method={method_name} epoch={epoch} train_loss={mean_loss:.4f} "
            f"seconds={epoch_seconds:.1f} images_per_second={images_per_second:.1f} "
            f"peak_gpu_memory_gb={peak_memory_gb:.2f}",
            flush=True,
        )

        checkpoint_data = {
            "model": model.state_dict(),
            "method": method_name,
            "encoder": encoder_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "training_signature": signature,
            "epoch": epoch,
            "best_loss": best_loss,
            "history": history,
            "rng_state": capture_rng_state(),
        }
        if mean_loss < best_loss:
            best_loss = mean_loss
            checkpoint_data["best_loss"] = best_loss
            atomic_torch_save(checkpoint_data, output_dir / "best.pt")
        atomic_torch_save(checkpoint_data, resume_path)

    save_json(
        output_dir / "metrics.json",
        {"method": method_name, "history": history, "device": str(device)},
    )
    finish_run(
        output_dir,
        epochs_completed=history[-1]["epoch"],
        training_seconds=sum(row["seconds"] for row in history),
        checkpoint_path=str(output_dir / "last.pt"),
        checkpoint_bytes=resume_path.stat().st_size,
    )


def main() -> None:
    train(load_config(parse_args().config))


if __name__ == "__main__":
    main()
