"""Train SimCLR on a prepared CheXpert pretraining manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from chexpert_ssl.data import SimCLRDataset, simclr_transform
from chexpert_ssl.losses import nt_xent_loss
from chexpert_ssl.models import SimCLRModel
from chexpert_ssl.utils import device_from_config, load_config, save_json, save_run_metadata, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    seed_everything(int(config["seed"]))
    device = device_from_config(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, config)

    frame = pd.read_csv(config["manifest"])
    dataset = SimCLRDataset(
        frame, simclr_transform(int(config["image_size"]), bool(config.get("horizontal_flip", False)))
    )
    samples = [dataset[index] for index in range(min(8, len(dataset)))]
    if samples:
        save_image(
            torch.stack([item for pair in samples for item in pair]),
            output_dir / "augmentation_pairs.png",
            normalize=True,
            nrow=2,
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = SimCLRModel(
        config.get("encoder", "resnet50"),
        int(config.get("projection_dim", 128)),
        pretrained=bool(config.get("imagenet_pretrained", False)),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    resume_path = output_dir / "last.pt"
    if bool(config.get("resume", True)) and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        history = checkpoint["history"]

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for first, second in loader:
            first, second = first.to(device, non_blocking=True), second.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = nt_xent_loss(model(first), model(second), float(config["temperature"]))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        mean_loss = total_loss / max(1, len(loader))
        history.append({"epoch": epoch, "train_loss": mean_loss, "learning_rate": scheduler.get_last_lr()[0]})
        print(f"epoch={epoch} train_loss={mean_loss:.4f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_loss": best_loss,
                "history": history,
            },
            resume_path,
        )

    save_json(output_dir / "metrics.json", {"history": history, "device": str(device)})


if __name__ == "__main__":
    main()
