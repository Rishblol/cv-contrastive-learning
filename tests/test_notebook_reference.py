"""Numerical parity with the user-supplied notebook, without running its data cells."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from chexpert_ssl.data import (
    TARGETS,
    nested_patient_ids,
    select_patient_cohort,
    split_by_patient,
    target_columns,
)
from chexpert_ssl.experiments import build_plan
from chexpert_ssl.gpu_augment import augment_grayscale_batch
from chexpert_ssl.metrics import select_thresholds, threshold_metrics
from chexpert_ssl.models import StandardizedHead
from chexpert_ssl.sampling import NotebookBatchSampler
from chexpert_ssl.ssl_methods import METHODS, build_ssl_method
from chexpert_ssl.utils import load_config
from scripts.run_experiments import notebook_smoke_suite
from scripts.train_downstream import positive_weights
from scripts.train_pretrain import notebook_learning_rate


def notebook_definitions(cell, names=None):
    notebook = json.loads(Path("chexpert_ssl_colab.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][cell]["source"]))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (names is None or node.name in names)
    ]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "copy": copy,
        "math": math,
        "torchvision": torchvision,
    }
    exec(  # noqa: S102 -- only selected definitions from the local reference; no data cells
        compile(ast.Module(body=definitions, type_ignores=[]), "reference-notebook", "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("method", METHODS)
def test_ssl_losses_and_encoder_gradients_match_notebook(method):
    torch.set_num_threads(1)
    reference = notebook_definitions(8)
    arguments = {
        "simclr": (0.2,),
        "moco": (0.2, 16, 0.99, 4),
        "byol": (0.99,),
        "nnclr": (0.1, 16),
        "swav": (0.1, 4),
    }
    names = {"simclr": "SimCLR", "moco": "MoCo", "byol": "BYOL", "nnclr": "NNCLR", "swav": "SwAV"}
    expected = reference[names[method]]("resnet18", *arguments[method], dim=8, hidden=16)
    actual = build_ssl_method(
        method,
        "resnet18",
        {
            "projection_dim": 8,
            "hidden_dim": 16,
            "temperatures": {method: 0.1 if method in {"nnclr", "swav"} else 0.2},
            "queue_size": 16,
            "momentum": 0.99,
            "bn_splits": 4,
            "positive_as_anchor": True,
            "prototype_count": 4,
            "prototype_freeze_steps": 0,
        },
    )
    state = {}
    substitutions = [
        ("k_encoder.net.", "key_encoder."),
        ("t_encoder.net.", "target_encoder."),
        ("encoder.net.", "encoder."),
        ("k_proj.", "key_projector."),
        ("t_proj.", "target_projector."),
        ("proj.", "projector."),
        ("pred.", "predictor."),
    ]
    for key, value in expected.state_dict().items():
        for old, new in substitutions:
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        key = {
            "ptr": "queue_pointer" if method == "moco" else "support_pointer",
            "filled": "support_filled",
        }.get(key, key)
        state[key] = value
    actual.load_state_dict(state)
    inputs = [torch.rand(4, 1, 64, 64) for _ in range(2)]
    torch.manual_seed(31)
    expected_loss = expected(*inputs)
    torch.manual_seed(31)
    actual_loss = actual(*[((x - 0.449) / 0.226).expand(-1, 3, -1, -1) for x in inputs])
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-5)
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(
        actual.encoder.conv1.weight.grad,
        expected.encoder.net.conv1.weight.grad,
        rtol=2e-3,
        atol=1e-4,
    )


def test_gpu_augmentation_and_lr_match_reference():
    reference = notebook_definitions(7, {"_u", "gpu_augment"})
    config = load_config(Path("configs/experiments/notebook_reference.yaml"))
    actual_config = build_plan(config)["pretrain"][0]["augmentation"]
    aliases = {
        "rotation": "rot",
        "translation": "trans",
        "blur_probability": "blur_p",
        "noise_probability": "noise_p",
        "horizontal_flip": "hflip",
    }
    expected_config = {
        aliases.get(k, k): v for k, v in actual_config.items() if k != "normalization"
    }
    images = torch.randint(0, 256, (4, 32, 32), dtype=torch.uint8)
    torch.manual_seed(19)
    expected = (reference["gpu_augment"](images, 32, expected_config) - 0.449) / 0.226
    torch.manual_seed(19)
    actual = augment_grayscale_batch(images, 32, actual_config, channels_last=False)
    torch.testing.assert_close(actual, expected.expand(-1, 3, -1, -1))
    lr = notebook_definitions(9, {"lr_at"})["lr_at"]
    for step in [0, 1, 19, 20, 57, 299]:
        assert notebook_learning_rate(step, 300, 20, 0.001) == lr(step, 300, 20, 0.001)


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
    reference = notebook_definitions(13, {"youden_thresholds"})
    from sklearn.metrics import roc_curve

    reference["youden_thresholds"].__globals__["roc_curve"] = roc_curve
    assert list(select_thresholds(y, p, "youden").values()) == reference["youden_thresholds"](y, p)
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
    monkeypatch.setattr("scripts.run_experiments.file_hash", lambda _: "input-hash")
    suite, root, _, source = notebook_smoke_suite(matrix)
    assert source["label_fraction"] == 0.05
    assert {c["method"] for name, c in suite.items() if name.startswith("pretrain")} == set(METHODS)
    assert len(suite) == 9
    assert suite["downstream-moco"]["ssl_checkpoint"] == str(root / "pretrain-moco" / "last.pt")
    assert all(c["epochs"] == 2 for c in suite.values())


def test_notebook_single_class_threshold_metrics_and_ceiling_cohorts(tmp_path):
    reference = notebook_definitions(13, {"compute_metrics"})
    from sklearn.metrics import average_precision_score, roc_auc_score

    reference["compute_metrics"].__globals__.update(
        LABELS=TARGETS, roc_auc_score=roc_auc_score, average_precision_score=average_precision_score
    )
    y = np.zeros((4, 5))
    y[:, 1] = 1
    p = np.tile([[0.1], [0.9], [0.1], [0.9]], (1, 5))
    expected = reference["compute_metrics"](y, p, [0.5] * 5)["threshold_metrics"]
    actual = threshold_metrics(y, p, dict.fromkeys(TARGETS, 0.5), notebook=True)
    for label, metrics in expected.items():
        for key, value in metrics.items():
            if key != "threshold":
                np.testing.assert_allclose(actual[f"{key}_{label}"], value, equal_nan=True)
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
    from chexpert_ssl.utils import save_json
    from scripts.aggregate_results import collect_results, write_label_efficiency_plot

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
