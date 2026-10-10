from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chexpert_ssl.experiments import build_plan
from chexpert_ssl.utils import config_signature, load_config, save_json
from scripts import run_experiments


@pytest.mark.parametrize(
    ("stage", "command_count"), [("prepare", 1), ("smoke", 5), ("aggregate", 1)]
)
def test_support_stages_ignore_unrelated_stale_experiment_plans(
    tmp_path, monkeypatch, stage, command_count
):
    matrix_path = tmp_path / "matrix.json"
    root = tmp_path / "plans"
    legacy = root / "downstream" / "stale.json"
    save_json(legacy, {"learning_rate": "1e-05"})
    before = legacy.read_bytes()
    save_json(matrix_path, {"generated_dir": str(root), "data_config": "configs/data/prepare.yaml"})
    calls = []
    monkeypatch.setattr(
        sys, "argv", ["run_experiments", "--config", str(matrix_path), "--stage", stage]
    )
    monkeypatch.setattr(
        run_experiments.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )

    def unexpected_plan(_):
        pytest.fail("Support stages must not build or materialize the experiment matrix")

    monkeypatch.setattr(run_experiments, "build_plan", unexpected_plan)
    monkeypatch.setattr(
        run_experiments,
        "smoke_settings",
        lambda _: (
            {"output_dir": str(tmp_path / "smoke" / "pretrain")},
            {"output_dir": str(tmp_path / "smoke" / "downstream")},
            tmp_path / "smoke",
            "source.csv",
        ),
    )
    if stage == "smoke":
        matrix = load_config(matrix_path)
        matrix["downstream_templates"] = {"ssl_linear": "source-config.yaml"}
        save_json(matrix_path, matrix)
    run_experiments.main()
    assert len(calls) == command_count
    assert legacy.read_bytes() == before
    assert not (root / "plan.json").exists()


def test_plan_reuses_legacy_numeric_string_without_rewriting_it(tmp_path, monkeypatch):
    matrix = load_config(Path("configs/experiments/label_efficiency.yaml"))
    matrix["generated_dir"] = str(tmp_path / "plans")
    plan = build_plan(matrix)
    config = next(c for c in plan["downstream"] if c["mode"] == "ssl_finetune")
    saved = {**config, "learning_rate": "1e-05"}
    path = Path(matrix["generated_dir"]) / "downstream" / f"{Path(config['output_dir']).name}.json"
    save_json(path, saved)
    before = path.read_bytes()
    matrix_path = tmp_path / "matrix.json"
    save_json(matrix_path, matrix)
    monkeypatch.setattr(
        sys, "argv", ["run_experiments", "--config", str(matrix_path), "--stage", "plan"]
    )
    run_experiments.main()
    assert path.read_bytes() == before
    assert load_config(path)["learning_rate"] == 0.00001
    assert load_config(Path(matrix["generated_dir"]) / "plan.json")["counts"]["downstream"] == 180


def test_numeric_equivalent_configs_have_the_same_resume_signature():
    assert config_signature({"learning_rate": "1e-05", "batch_size": "64"}) == config_signature(
        {"learning_rate": 0.00001, "batch_size": 64}
    )


def test_smoke_output_versions_are_stable_and_preserve_legacy_paths(monkeypatch):
    matrix = load_config(Path("configs/experiments/label_efficiency.yaml"))
    legacy = load_config(Path("configs/pretrain/simclr_smoke.yaml"))["output_dir"]
    monkeypatch.setattr(run_experiments, "file_hash", lambda _: "unchanged-input")
    monkeypatch.setattr(run_experiments, "implementation_hash", lambda _: "implementation-one")
    first = run_experiments.smoke_settings(matrix)
    second = run_experiments.smoke_settings(matrix)
    assert first == second
    assert first[0]["output_dir"] != legacy
    assert first[1]["ssl_checkpoint"] == str(Path(first[0]["output_dir"]) / "best.pt")
    assert first[0]["resume"] and first[1]["resume"]
    monkeypatch.setattr(run_experiments, "implementation_hash", lambda _: "implementation-two")
    assert run_experiments.smoke_settings(matrix)[2] != first[2]
    assert load_config(Path("configs/pretrain/simclr_smoke.yaml"))["output_dir"] == legacy
