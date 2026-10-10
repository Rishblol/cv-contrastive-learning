from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from chexpert_ssl.data import TARGETS, read_manifest, select_patient_cohort
from chexpert_ssl.experiments import build_plan
from chexpert_ssl.models import MultiLabelClassifier
from chexpert_ssl.selection import lock_run, verify_selection
from chexpert_ssl.ssl_methods import METHODS
from chexpert_ssl.utils import (
    artifact_lock,
    capture_rng_state,
    config_signature,
    load_config,
    restore_rng_state,
    save_json,
)
from scripts.aggregate_results import collect_results
from scripts.evaluate_checkpoint import evaluate
from scripts.train_downstream import train as train_downstream
from scripts.train_pretrain import train as train_pretrain


def write_records(root: Path, partition: str, start: int, patients: int) -> Path:
    rows = []
    for patient in range(start, start + patients):
        for observation in (0, 1):
            path = root / partition / f"patient{patient:05d}" / "study1" / f"view{observation}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(patient * 2 + observation)
            Image.fromarray(rng.integers(0, 256, (40, 40), dtype=np.uint8)).save(path)
            rows.append(
                {
                    "image_path": str(path),
                    "Path": str(path),
                    "patient_id": f"{patient:05d}",
                    "study_id": "0001",
                    "Frontal/Lateral": "Frontal",
                    "AP/PA": "AP",
                    "Age": 50,
                    "Sex": "Female",
                    **{f"target_{label}": observation for label in TARGETS},
                }
            )
    manifest = root / f"{partition}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


@pytest.fixture
def synthetic_data(tmp_path):
    train = write_records(tmp_path, "train", 1, 2)
    dev = write_records(tmp_path, "development", 101, 2)
    heldout = write_records(tmp_path, "heldout", 201, 2)
    cohort = tmp_path / "cohort.json"
    save_json(cohort, {"seed": 42, "label_fraction": 1.0, "patient_ids": ["00001", "00002"]})
    return train, dev, heldout, cohort


def pretrain_config(tmp_path, synthetic_data, method):
    train, dev, _, _ = synthetic_data
    config = load_config(Path(f"configs/pretrain/{method}.yaml"))
    config.update(
        manifest=str(train),
        development_manifest=str(dev),
        device="cpu",
        precision="float32",
        image_size=32,
        batch_size=2,
        epochs=1,
        num_workers=0,
        cache_workers=1,
        torch_num_threads=1,
        output_dir=str(tmp_path / method),
        cache_path=str(tmp_path / "images.npy"),
        resume=True,
    )
    config["ssl"] = {
        "hidden_dim": 16,
        "projection_dim": 8,
        "queue_size": 4,
        "prototype_count": 4,
        "prototype_freeze_steps": 0,
    }
    return config


def downstream_config(tmp_path, synthetic_data, mode="supervised", checkpoint=None):
    train, dev, _, cohort = synthetic_data
    config = load_config(Path("configs/downstream/supervised.yaml"))
    config.update(
        train_manifest=str(train),
        development_manifest=str(dev),
        sampled_patients=str(cohort),
        label_fraction=1.0,
        image_size=32,
        batch_size=2,
        epochs=1,
        num_workers=0,
        torch_num_threads=1,
        mode=mode,
        output_dir=str(tmp_path / mode),
        device="cpu",
        precision="float32",
    )
    if checkpoint:
        config.update(ssl_method="simclr", ssl_checkpoint=str(checkpoint))
    return config


def test_manifest_cohort_roundtrip_preserves_identifiers(synthetic_data):
    train, _, _, cohort = synthetic_data
    records = read_manifest(train)
    assert records.patient_id.tolist() == ["00001", "00001", "00002", "00002"]
    assert records.study_id.iloc[0] == "0001"
    assert len(select_patient_cohort(records, cohort, 1.0, 42)) == 4
    with pytest.raises(ValueError, match="seed/budget"):
        select_patient_cohort(records, cohort, 1.0, 43)
    save_json(cohort, {"seed": 42, "label_fraction": 1.0, "patient_ids": ["00001", "99999"]})
    with pytest.raises(ValueError, match="unknown"):
        select_patient_cohort(records, cohort, 1.0, 42)


def test_config_inheritance_merges_and_rejects_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("nested: {a: 1, b: 2}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nnested: {b: 3}\n")
    assert load_config(tmp_path / "child.yaml") == {"nested": {"a": 1, "b": 3}}
    (tmp_path / "base.yaml").write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "child.yaml")


def test_full_matrix_has_five_pretrains_and_180_matched_runs():
    matrix = load_config(Path("configs/experiments/label_efficiency.yaml"))
    plan = build_plan(matrix)
    assert {config["method"] for config in plan["pretrain"]} == set(METHODS)
    assert len(plan["downstream"]) == len(plan["evaluate"]) == 180
    for seed in matrix["seeds"]:
        for fraction in matrix["budgets"]:
            pair = [
                c
                for c in plan["downstream"]
                if c["seed"] == seed and c["label_fraction"] == fraction
            ]
            assert len(pair) == 12
            assert len({c["sampled_patients"] for c in pair}) == 1
    broken = copy.deepcopy(matrix)
    broken["pretrain_configs"].pop("swav")
    with pytest.raises(ValueError, match="primary matrix"):
        build_plan(broken)


@pytest.mark.parametrize("method", METHODS)
def test_each_ssl_pretraining_writes_and_resumes_artifacts(
    tmp_path, synthetic_data, method, monkeypatch
):
    config = pretrain_config(tmp_path, synthetic_data, method)
    train_pretrain(config)
    output = Path(config["output_dir"])
    assert (output / "augmentation_pairs.png").exists()
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    assert checkpoint["method"] == method
    assert np.isfinite(checkpoint["history"][0]["train_loss"])
    assert checkpoint["encoder"]
    train_pretrain(config)
    changed = {**config, "learning_rate": 0.1}
    before = (output / "run_metadata.json").read_bytes()
    with pytest.raises(ValueError, match="Checkpoint settings"):
        train_pretrain(changed)
    assert (output / "run_metadata.json").read_bytes() == before
    import chexpert_ssl.utils as utilities

    monkeypatch.setattr(utilities, "implementation_hash", lambda _: "changed-implementation")
    with pytest.raises(ValueError, match="Checkpoint settings"):
        train_pretrain(config)
    assert (output / "run_metadata.json").read_bytes() == before


@pytest.mark.parametrize("mode", ["supervised", "ssl_linear", "ssl_finetune"])
def test_downstream_checkpoint_lock_evaluate_and_aggregate(tmp_path, synthetic_data, mode):
    source = None
    if mode != "supervised":
        config = pretrain_config(tmp_path, synthetic_data, "simclr")
        train_pretrain(config)
        source = Path(config["output_dir"]) / "best.pt"
    config = downstream_config(tmp_path, synthetic_data, mode, source)
    train_downstream(config)
    run = Path(config["output_dir"])
    selected = torch.load(run / "best.pt", weights_only=False)
    model = MultiLabelClassifier("resnet18", 5).eval()
    model.load_state_dict(selected["model"])
    sample = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(sample)
    reloaded = MultiLabelClassifier("resnet18", 5).eval()
    reloaded.load_state_dict(torch.load(run / "best.pt", weights_only=False)["model"])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(sample), expected, rtol=0, atol=0)
    train_downstream(config)
    changed = {**config, "learning_rate": 0.1}
    with pytest.raises(ValueError, match="Resume configuration"):
        train_downstream(changed)
    lock = lock_run(run, "Synthetic development smoke selection")
    evaluation = {
        "checkpoint": str(run / "best.pt"),
        "thresholds": str(run / "thresholds.json"),
        "selection_lock": str(lock),
        "manifest": str(synthetic_data[2]),
        "output_dir": str(tmp_path / "reports" / mode),
        "batch_size": 2,
        "seed": 42,
        "bootstrap_samples": 5,
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="conflicts"):
        verify_selection({**evaluation, "image_size": 224}, selected)
    evaluate(evaluation)
    final = collect_results(tmp_path, "final_validation")
    dev = collect_results(tmp_path, "development")
    assert len(final) == len(dev) == 1
    assert final.iloc[0].method == dev.iloc[0].method
    assert final.iloc[0].label_fraction == 1.0
    predictions = pd.read_csv(Path(evaluation["output_dir"]) / "predictions.csv")
    assert {"AP/PA", "Age", "Sex"}.issubset(predictions)
    with pytest.raises(ValueError, match="already exist"):
        evaluate(evaluation)
    with pytest.raises(ValueError, match="immutable"):
        train_downstream(config)
    save_json(run / "thresholds.json", {label: 0.123 for label in TARGETS})
    with pytest.raises(ValueError, match="changed after locking"):
        verify_selection(evaluation, selected)


def test_training_rejects_development_patient_leakage(tmp_path, synthetic_data):
    config = pretrain_config(tmp_path, synthetic_data, "simclr")
    config["development_manifest"] = config["manifest"]
    with pytest.raises(ValueError, match="leakage"):
        train_pretrain(config)
    downstream = downstream_config(tmp_path, synthetic_data)
    downstream["development_manifest"] = downstream["train_manifest"]
    with pytest.raises(ValueError, match="leakage"):
        train_downstream(downstream)


def test_rng_state_restoration_reproduces_next_draws():
    state = capture_rng_state()
    expected = torch.rand(4), np.random.random(4)
    restore_rng_state(state)
    actual = torch.rand(4), np.random.random(4)
    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
    np.testing.assert_array_equal(expected[1], actual[1])


def test_signature_ignores_output_and_resume_but_tracks_training_choices():
    config = {"output_dir": "a", "resume": True, "learning_rate": 0.1}
    assert config_signature(config) == config_signature(
        {**config, "output_dir": "b", "resume": False}
    )
    assert config_signature(config) != config_signature({**config, "learning_rate": 0.2})


def test_artifact_lock_rejects_concurrent_writers_and_releases(tmp_path):
    path = tmp_path / "lock"
    with (
        artifact_lock(path),
        pytest.raises(RuntimeError, match="Another process"),
        artifact_lock(path),
    ):
        pass
    with artifact_lock(path):
        pass


@pytest.mark.parametrize("method", ["simclr", "moco"])
def test_interrupted_pretraining_matches_uninterrupted_run(
    tmp_path, synthetic_data, monkeypatch, method
):
    import scripts.train_pretrain as entry

    config = pretrain_config(tmp_path, synthetic_data, method)
    config["epochs"] = 2
    entry.train(config)
    expected = torch.load(Path(config["output_dir"]) / "last.pt", weights_only=False)["model"]
    interrupted = {**config, "output_dir": str(tmp_path / "interrupted")}
    original_save = entry.atomic_torch_save

    def interrupt_after_checkpoint(payload, path):
        original_save(payload, path)
        if path.name == "last.pt" and payload["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(entry, "atomic_torch_save", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        entry.train(interrupted)
    assert load_config(Path(interrupted["output_dir"]) / "run_metadata.json")["status"] == "failed"
    monkeypatch.setattr(entry, "atomic_torch_save", original_save)
    entry.train(interrupted)
    actual = torch.load(Path(interrupted["output_dir"]) / "last.pt", weights_only=False)["model"]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_interrupted_downstream_matches_uninterrupted_run(tmp_path, synthetic_data, monkeypatch):
    import scripts.train_downstream as entry

    config = downstream_config(tmp_path, synthetic_data)
    config.update(epochs=2, num_workers=1)
    entry.train(config)
    expected = torch.load(Path(config["output_dir"]) / "last.pt", weights_only=False)["model"]
    interrupted = {**config, "output_dir": str(tmp_path / "interrupted")}
    original_save = entry.atomic_torch_save

    def interrupt_after_checkpoint(payload, path):
        original_save(payload, path)
        if path.name == "last.pt" and payload["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(entry, "atomic_torch_save", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        entry.train(interrupted)
    monkeypatch.setattr(entry, "atomic_torch_save", original_save)
    entry.train(interrupted)
    actual = torch.load(Path(interrupted["output_dir"]) / "last.pt", weights_only=False)["model"]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_imagenet_training_pins_weights_and_evaluation_never_downloads(
    tmp_path, synthetic_data, monkeypatch
):
    from torchvision import models

    constructor = models.resnet18
    requested = []

    def local_constructor(*, weights):
        requested.append(weights)
        return constructor(weights=None)

    monkeypatch.setattr(models, "resnet18", local_constructor)
    config = downstream_config(tmp_path, synthetic_data)
    config["imagenet_pretrained"] = True
    train_downstream(config)
    run = Path(config["output_dir"])
    lock = lock_run(run, "Synthetic ImageNet lifecycle test")
    evaluate(
        {
            "checkpoint": str(run / "best.pt"),
            "thresholds": str(run / "thresholds.json"),
            "selection_lock": str(lock),
            "manifest": str(synthetic_data[2]),
            "output_dir": str(tmp_path / "evaluation"),
            "batch_size": 2,
            "seed": 42,
            "bootstrap_samples": 5,
            "device": "cpu",
        }
    )
    assert requested == [models.ResNet18_Weights.IMAGENET1K_V1, None]
    final = collect_results(tmp_path, "final_validation")
    assert final.iloc[0].method == "imagenet_supervised"
    assert final.iloc[0].pretraining_modality == "ImageNet"


def test_prepare_data_config_produces_canonical_manifests_and_nested_cohorts(tmp_path, monkeypatch):
    import sys

    from scripts.prepare_data import main

    root = tmp_path / "dataset"
    root.mkdir()
    for partition, start, count in [("train", 1, 5), ("valid", 101, 2)]:
        manifest = write_records(root, partition, start, count)
        frame = pd.read_csv(manifest, dtype={"patient_id": str})
        for label in TARGETS:
            frame[label] = frame[f"target_{label}"]
        if partition == "train":
            frame.loc[frame.patient_id.eq("00001"), "Frontal/Lateral"] = "Lateral"
        frame.to_csv(manifest, index=False)
    output = tmp_path / "processed" / "manifests"
    config = tmp_path / "prepare.json"
    save_json(
        config,
        {
            "dataset_root": str(root),
            "output_dir": str(output),
            "dev_fraction": 0.4,
            "seed": 42,
            "frontal_only_downstream": True,
            "budgets": [0.5, 1.0],
            "budget_seeds": [42, 43],
        },
    )
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--config", str(config)])
    main()
    pretrain = read_manifest(output / "pretrain_train_leakage_free.csv", labeled=False)
    assert (output / "pretrain_train.csv").read_bytes() == (
        output / "pretrain_train_leakage_free.csv"
    ).read_bytes()
    splits = output.parent / "splits"
    train_ids = set(pd.read_csv(splits / "train_patients.csv", dtype=str).patient_id)
    dev_ids = set(pd.read_csv(splits / "development_patients.csv", dtype=str).patient_id)
    assert train_ids == set(pretrain.patient_id)
    assert not train_ids & dev_ids
    assert train_ids | dev_ids == {f"{value:05d}" for value in range(1, 6)}
    for seed in [42, 43]:
        small = load_config(splits / f"budget_0p5_seed_{seed}.json")
        full = load_config(splits / f"budget_1_seed_{seed}.json")
        assert set(small["patient_ids"]).issubset(full["patient_ids"])
        select_patient_cohort(
            read_manifest(output / "downstream_train.csv"),
            splits / f"budget_1_seed_{seed}.json",
            1.0,
            seed,
        )
    with pytest.raises(ValueError, match="already exists"):
        main()


def test_aggregation_separates_partitions_and_rejects_duplicate_pairs(tmp_path):
    config = {
        "mode": "supervised",
        "encoder": "resnet18",
        "seed": 42,
        "label_fraction": 0.01,
        "imagenet_pretrained": False,
    }
    for name, partition, score in [
        ("development", "development", 0.2),
        ("reports/final", "final_validation", 0.8),
    ]:
        directory = tmp_path / name
        save_json(directory / "metrics.json", {"partition": partition, "macro_auroc": score})
        save_json(directory / "run_metadata.json", {"status": "complete", "config": config})
    assert collect_results(tmp_path, "development").iloc[0].macro_auroc == 0.2
    assert collect_results(tmp_path, "final_validation").iloc[0].macro_auroc == 0.8
    save_json(
        tmp_path / "duplicate" / "metrics.json",
        {"partition": "final_validation", "macro_auroc": 0.9},
    )
    save_json(
        tmp_path / "duplicate" / "run_metadata.json", {"status": "complete", "config": config}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        collect_results(tmp_path, "final_validation")
