"""Scientific protocol regressions for the consolidated notebook."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from notebook_runtime import (
    METHODS,
    TARGETS,
    NotebookBatchSampler,
    StandardizedHead,
    build_plan,
    load_config,
    nested_patient_ids,
    notebook_smoke_suite,
    positive_weights,
    select_patient_cohort,
    select_thresholds,
    split_by_patient,
    target_columns,
    threshold_metrics,
)
from torch.nn import functional as F


def test_reference_plan_and_patient_sampling():
    matrix = load_config(Path("configs/experiments/notebook_reference.yaml"))
    plan = build_plan(matrix)
    assert {k: len(v) for k, v in plan.items()} == {
        "pretrain": 5,
        "downstream": 180,
        "evaluate": 180,
    }
    assert matrix["seeds"] == [0, 1, 2]
    assert [c["imagenet_pretrained"] for c in plan["downstream"][:2]] == [False, True]
    assert [c["mode"] for c in plan["downstream"][2:7]] == ["ssl_linear"] * 5
    assert [c["mode"] for c in plan["downstream"][7:12]] == ["ssl_finetune"] * 5
    assert all(c["batch_size"] == 256 and c["ssl"]["hidden_dim"] == 1024 for c in plan["pretrain"])
    for config in plan["downstream"]:
        linear = config["mode"] == "ssl_linear"
        assert config["epochs"] == (
            100 if linear else matrix["epochs_by_budget"][config["label_fraction"]]
        )
        assert config["batch_size"] == (512 if linear else 64)
        if config.get("ssl_method"):
            assert config["ssl_checkpoint"].endswith("last.pt")
    frame = pd.DataFrame({"patient_id": [f"{i:05d}" for i in range(23)]})
    train, dev = split_by_patient(frame, 0.1, 1234, random_state=True)
    order = np.random.RandomState(1234).permutation(frame.patient_id.to_numpy())
    assert set(dev.patient_id) == set(order[:2])
    for seed in matrix["seeds"]:
        samples = nested_patient_ids(train, matrix["budgets"], seed, True, "ceil")
        ordered = np.random.RandomState(seed).permutation(sorted(train.patient_id))
        for fraction, patients in samples.items():
            assert patients == ordered[: max(1, math.ceil(len(ordered) * fraction))].tolist()


def test_probe_standardization_weights_and_youden_match_notebook():
    features = torch.randn(12, 8)
    head = StandardizedHead(8, 5)
    head.mean.copy_(features.mean(0))
    head.std.copy_(features.std(0).clamp_min(1e-6))
    folded = F.linear(
        features,
        head.linear.weight / head.std,
        head.linear.bias - (head.linear.weight * head.mean / head.std).sum(1),
    )
    torch.testing.assert_close(head(features), folded)
    frame = pd.DataFrame(
        {column: [1, 1, 1, 1] if i == 0 else [0] * 4 for i, column in enumerate(target_columns())}
    )
    torch.testing.assert_close(positive_weights(frame, 50), torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0]))
    y = np.tile([[0], [0], [1], [1]], (1, 5))
    p = np.tile([[0.1], [0.4], [0.35], [0.8]], (1, 5))
    assert list(select_thresholds(y, p, "youden").values()) == [0.8] * 5
    assert len(TARGETS) == 5


def test_notebook_epoch_batch_orders_and_smoke_wiring(monkeypatch):
    records, batch, seed = 23, 4, 2
    for protocol in ("ssl", "finetune", "linear"):
        sampler = NotebookBatchSampler(records, batch, seed, protocol)
        sampler.epoch = 3
        if protocol == "ssl":
            order = np.random.RandomState(seed * 1000 + 3).permutation(records)
        elif protocol == "finetune":
            rng = np.random.RandomState(seed)
            for _ in range(4):
                order = rng.permutation(records)
        else:
            generator = torch.Generator().manual_seed(seed)
            for _ in range(4):
                order = torch.randperm(records, generator=generator).numpy()
        expected = []
        for start in range(0, records, batch):
            chunk = order[start : start + batch]
            if protocol == "ssl" and len(chunk) != batch:
                continue
            expected.append((chunk if protocol == "linear" else np.sort(chunk)).tolist())
        assert list(sampler) == expected
    matrix = load_config(Path("configs/experiments/notebook_reference.yaml"))
    monkeypatch.setattr("notebook_runtime.file_hash", lambda _: "input-hash")
    suite, root, _, source = notebook_smoke_suite(matrix)
    assert source["label_fraction"] == 0.05
    assert {c["method"] for name, c in suite.items() if name.startswith("pretrain")} == set(METHODS)
    assert len(suite) == 9
    assert suite["downstream-moco"]["ssl_checkpoint"] == str(root / "pretrain-moco" / "last.pt")
    assert all(c["epochs"] == 2 for c in suite.values())


def test_notebook_single_class_threshold_metrics_and_ceiling_cohorts(tmp_path):
    y = np.zeros((4, 5))
    y[:, 1] = 1
    p = np.tile([[0.1], [0.9], [0.1], [0.9]], (1, 5))
    actual = threshold_metrics(y, p, dict.fromkeys(TARGETS, 0.5), notebook=True)
    assert actual["specificity_Atelectasis"] == 0.5
    assert actual["sensitivity_Cardiomegaly"] == 0.5
    assert np.isnan(actual["sensitivity_Atelectasis"])
    frame = pd.DataFrame({"patient_id": [str(i) for i in range(21)]})
    cohort = tmp_path / "cohort.json"
    payload = {
        "seed": 0,
        "label_fraction": 0.1,
        "budget_rounding": "ceil",
        "patient_ids": ["0", "1", "2"],
    }
    cohort.write_text(json.dumps(payload))
    assert len(select_patient_cohort(frame, cohort, 0.1, 0)) == 3
    payload.pop("budget_rounding")
    cohort.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="count"):
        select_patient_cohort(frame, cohort, 0.1, 0)


def test_reference_report_export_and_smoke_exclusion(tmp_path):
    from notebook_runtime import collect_results, save_json, write_label_efficiency_plot

    config = {
        "mode": "supervised",
        "encoder": "resnet18",
        "seed": 0,
        "label_fraction": 0.01,
        "imagenet_pretrained": False,
    }
    for folder, score in [("downstream/run", 0.7), ("smoke/run", 0.9), ("archive/run", 0.8)]:
        directory = tmp_path / folder
        save_json(directory / "metrics.json", {"partition": "development", "macro_auroc": score})
        save_json(directory / "run_metadata.json", {"status": "complete", "config": config})
    results = collect_results(tmp_path, "development")
    assert len(results) == 1 and results.macro_auroc.iloc[0] == 0.7
    write_label_efficiency_plot(results, tmp_path)
    assert (tmp_path / "label_efficiency.png").stat().st_size > 0
    assert (tmp_path / "macro_auroc_table.csv").is_file()
