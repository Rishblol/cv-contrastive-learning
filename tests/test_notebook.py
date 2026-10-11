"""Integration checks for the notebook as the sole executable implementation."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import nbformat
import notebook_runtime as runtime
import numpy as np
import pytest
import torch
from nbclient import NotebookClient


@pytest.mark.parametrize("start_directory", [".", "tests"])
def test_notebook_schema_and_setup_in_jupyter(start_directory):
    notebook = nbformat.read(runtime.NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.metadata.get("tags")
    # Test setup in a fresh kernel without starting the full real-data study.
    notebook.cells = runtime.setup_cells(notebook)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert Path.cwd() == PROJECT_ROOT\n"
            "assert NOTEBOOK_PATH == PROJECT_ROOT / 'chexpert_ssl.ipynb'\n"
        )
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(runtime.PROJECT_ROOT / start_directory)}},
    ).execute()
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")


def test_definition_cells_do_not_import_removed_package():
    notebook = json.loads(runtime.NOTEBOOK_PATH.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("chexpert_ssl", "scripts"))
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith(("chexpert_ssl", "scripts")) for alias in node.names
                )
    for directory in ("src", "scripts", "configs"):
        assert not (runtime.PROJECT_ROOT / directory).exists()


def test_embedded_config_inheritance_without_external_files():
    matrix = runtime.load_config("configs/experiments/notebook_reference.yaml")
    plan = runtime.build_plan(matrix)
    assert len(plan["pretrain"]) == 5
    assert len(plan["downstream"]) == len(plan["evaluate"]) == 180
    assert all(config["num_workers"] == 0 for config in plan["pretrain"] + plan["downstream"])
    assert plan["pretrain"][0]["augmentation"]["normalization"] == {"mean": 0.449, "std": 0.226}
    assert len({c["sampled_patients"] for c in plan["downstream"]}) == 15


def test_explicit_task_arguments_ignore_kernel_argv(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.sys, "argv", ["ipykernel_launcher", "-f", "connection.json"])
    calls = []
    monkeypatch.setitem(
        runtime.TASKS, "prepare_data", lambda argv: calls.append((Path.cwd(), argv))
    )
    monkeypatch.chdir(tmp_path)
    runtime.execute_task(["prepare_data", "--config", "configs/data/notebook_reference.yaml"])
    assert calls == [(runtime.PROJECT_ROOT, ["--config", "configs/data/notebook_reference.yaml"])]
    assert Path.cwd() == tmp_path
    with pytest.raises(ValueError, match="Unknown notebook task"):
        runtime.execute_task(["unknown"])
    assert Path.cwd() == tmp_path


def test_run_stage_filters_and_working_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runtime, "run_experiments_cli", lambda argv: calls.append(argv))
    monkeypatch.chdir(tmp_path)
    runtime.run_stage("downstream", methods=["simclr"], budgets=[0.01], seeds=[0])
    assert calls[0][-6:] == ["--methods", "simclr", "--budgets", "0.01", "--seeds", "0"]
    assert Path.cwd() == tmp_path
    with pytest.raises(ValueError, match="nonempty"):
        runtime.run_stage("downstream", methods=[])


def test_implementation_hash_ignores_outputs_but_rejects_saved_code_changes(monkeypatch, tmp_path):
    notebook = json.loads(runtime.NOTEBOOK_PATH.read_text())
    path = tmp_path / "notebook.ipynb"
    original = runtime.implementation_hash()
    execution_cell = next(c for c in notebook["cells"] if c["id"] == "run-plan")
    execution_cell["execution_count"] = 12
    execution_cell["outputs"] = []
    path.write_text(json.dumps(notebook))
    monkeypatch.setattr(runtime, "NOTEBOOK_PATH", path)
    assert runtime.implementation_hash() == original
    library = next(c for c in notebook["cells"] if c["id"] == "models")
    library["source"].append("\n# saved implementation edit\n")
    path.write_text(json.dumps(notebook))
    with pytest.raises(RuntimeError, match="restart"):
        runtime.implementation_hash()


def test_notebook_rejects_spawn_workers():
    config = runtime.load_config("configs/pretrain/simclr.yaml")
    config["num_workers"] = 1
    with pytest.raises(ValueError, match="num_workers=0"):
        runtime.validate_training_config(config, "pretrain")


def test_learning_rate_boundaries_and_midpoint():
    assert runtime.notebook_learning_rate(0, 300, 20, 0.001) == 0.001 / 20
    assert runtime.notebook_learning_rate(19, 300, 20, 0.001) == 0.001
    assert runtime.notebook_learning_rate(20, 300, 20, 0.001) == 0.001
    assert runtime.notebook_learning_rate(160, 300, 20, 0.001) == pytest.approx(0.0005005)
    assert runtime.notebook_learning_rate(300, 300, 20, 0.001) == pytest.approx(0.000001)


def test_nt_xent_matches_manual_positive_probabilities():
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    second = first.detach().clone().requires_grad_()
    loss = runtime.nt_xent_loss(first, second, temperature=0.5)
    expected = math.log(math.exp(2) + 2) - 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()


def reviewed_vlm_config():
    return {
        "checkpoint": "example/reviewed-radiology-model",
        "revision": "a" * 40,
        "model_card_url": "https://example.org/model-card",
        "license_review": "Approved for research",
        "overlap_review": "Documented in study provenance",
        "radiology_domain_reviewed": True,
        "pretraining_modality": "image-text VLM",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("revision", "main"),
        ("checkpoint", "REPLACE_WITH_PINNED_CHECKPOINT"),
        ("model_card_url", ""),
        ("license_review", "required-before-run"),
        ("overlap_review", ""),
        ("radiology_domain_reviewed", False),
        ("pretraining_modality", "image-only SSL"),
    ],
)
def test_vlm_rejects_unreviewed_configuration(key, value):
    config = reviewed_vlm_config()
    runtime.validate_vlm_config(config)
    config[key] = value
    with pytest.raises(ValueError):
        runtime.validate_vlm_config(config)


def test_vlm_frozen_encoder_remains_in_eval_mode():
    class Encoder(torch.nn.Module):
        feature_dim = 3

        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)

        def image_features(self, values):
            return self.bn(values)

    encoder = Encoder()
    model = runtime.VLMClassifier(encoder, 5, freeze_encoder=True)
    before = copy.deepcopy(encoder.state_dict())
    model.train()
    model(torch.randn(4, 3)).sum().backward()
    assert not encoder.training
    assert model.classifier.weight.grad is not None
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_vlm_prompt_scoring_uses_normalized_two_class_similarity():
    concepts = {
        label: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) for label in runtime.TARGETS
    }
    scores = runtime.zero_shot_scores(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), concepts)
    np.testing.assert_allclose(scores[0], 1 / (1 + math.exp(-1)), rtol=1e-6)
    np.testing.assert_allclose(scores[1], 1 / (1 + math.exp(1)), rtol=1e-6)


@pytest.mark.parametrize("task", sorted(runtime.TASKS))
def test_all_task_parsers_accept_explicit_help(task, monkeypatch):
    monkeypatch.setattr(runtime.sys, "argv", ["ipykernel_launcher", "-f", "connection.json"])
    with pytest.raises(SystemExit) as result:
        runtime.execute_task([task, "--help"])
    assert result.value.code == 0


def test_managed_calls_anchor_relative_outputs_and_restore_cwd(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(runtime, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)

    @runtime.managed_run
    def write_result(config):
        assert Path.cwd() == root
        runtime.save_json(Path(config["output_dir"]) / "result.json", {"ok": True})

    write_result({"output_dir": "outputs/check"})
    assert (root / "outputs/check/result.json").is_file()
    assert not (elsewhere / "outputs").exists()
    assert Path.cwd() == elsewhere


def test_root_override_and_nested_directory_discovery(monkeypatch, tmp_path):
    monkeypatch.delenv("CHEXPERT_PROJECT_ROOT", raising=False)
    assert runtime.find_project_root(runtime.PROJECT_ROOT / "tests") == runtime.PROJECT_ROOT
    monkeypatch.setenv("CHEXPERT_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No CheXpert notebook"):
        runtime.find_project_root()
    (tmp_path / "chexpert_ssl.ipynb").write_text("{}")
    assert runtime.find_project_root() == tmp_path


def test_simclr_loss_and_gradients_match_explicit_normalized_objective():
    torch.manual_seed(7)
    model = runtime.SimCLR("resnet18", 8, 16, 0.2).eval()
    first, second = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    expected_model = copy.deepcopy(model)
    z = torch.nn.functional.normalize(
        expected_model.projector(expected_model.encoder(torch.cat((first, second)))).float(), dim=1
    )
    logits = z @ z.T / 0.2
    logits.fill_diagonal_(float("-inf"))
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 3, 0, 1]))
    actual = model(first, second)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(
        model.encoder.conv1.weight.grad, expected_model.encoder.conv1.weight.grad, rtol=0, atol=0
    )


def test_jupyter_synthetic_training_and_locked_evaluation(tmp_path, monkeypatch):
    from test_pipeline import write_records

    monkeypatch.delenv("CHEXPERT_PROJECT_ROOT", raising=False)
    # A standalone copy exercises notebook-relative paths with no installed project
    # package. All data are synthetic and all artifacts remain under tmp_path.
    notebook_path = tmp_path / "chexpert_ssl.ipynb"
    notebook_path.write_bytes(runtime.NOTEBOOK_PATH.read_bytes())
    for partition, first in [("train", 1), ("development", 101), ("heldout", 201)]:
        write_records(tmp_path, partition, first, 2)
    runtime.save_json(
        tmp_path / "cohort.json",
        {"seed": 42, "label_fraction": 1.0, "patient_ids": ["00001", "00002"]},
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells = runtime.setup_cells(notebook)
    notebook.cells.append(
        nbformat.v4.new_code_cell("""
assert Path.cwd() == PROJECT_ROOT
pretrain_config = load_config("configs/pretrain/simclr.yaml")
pretrain_config.update(
    manifest="train.csv", development_manifest="development.csv",
    cache_path="data/processed/synthetic.npy", output_dir="outputs/pretrain",
    device="cpu", precision="float32", image_size=32, batch_size=2, epochs=1,
    num_workers=0, cache_workers=1, torch_num_threads=1,
    ssl={"hidden_dim": 16, "projection_dim": 8},
)
train_pretrain(pretrain_config)
downstream_config = load_config("configs/downstream/simclr_linear.yaml")
downstream_config.update(
    train_manifest="train.csv", development_manifest="development.csv",
    sampled_patients="cohort.json", label_fraction=1.0, seed=42,
    ssl_checkpoint="outputs/pretrain/last.pt", output_dir="outputs/downstream",
    device="cpu", precision="float32", image_size=32, batch_size=2, epochs=1,
    num_workers=0, torch_num_threads=1,
)
train_downstream(downstream_config)
lock_run(Path("outputs/downstream"), "Synthetic development selection in a Jupyter kernel")
evaluate({
    "checkpoint": "outputs/downstream/best.pt",
    "thresholds": "outputs/downstream/thresholds.json",
    "selection_lock": "outputs/downstream/selection_lock.json",
    "manifest": "heldout.csv", "output_dir": "outputs/evaluation",
    "device": "cpu", "seed": 42, "batch_size": 2, "num_workers": 0,
    "bootstrap_samples": 3,
})
assert len(collect_results(Path("outputs"), "development")) == 1
assert len(collect_results(Path("outputs"), "final_validation")) == 1
""")
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    assert (tmp_path / "outputs/pretrain/last.pt").is_file()
    assert (tmp_path / "outputs/downstream/selection_lock.json").is_file()
    assert (tmp_path / "outputs/evaluation/predictions.csv").is_file()
    assert (
        runtime.load_config(tmp_path / "outputs/evaluation/run_metadata.json")["status"]
        == "complete"
    )


def test_run_all_calls_every_stage_in_order_without_switches():
    notebook = nbformat.read(runtime.NOTEBOOK_PATH, as_version=4)
    boundary = next(i for i, cell in enumerate(notebook.cells) if cell.id == "study-start")
    notebook.cells.insert(
        boundary,
        nbformat.v4.new_code_cell("""
observed_stages = []
def run_stage(stage, **kwargs):
    observed_stages.append((stage, kwargs))
def prepare_or_reuse_data():
    observed_stages.append(("prepare", {}))
def review_augmentations():
    observed_stages.append(("review", {}))
def lock_study_selection():
    observed_stages.append(("lock", {}))
"""),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell("""
assert observed_stages == [
    ("plan", {}), ("prepare", {}), ("smoke", {}), ("review", {}),
    ("pretrain", {"augmentation_reviewed": True}), ("downstream", {}),
    ("aggregate", {"partition": "development"}), ("lock", {}),
    ("evaluate", {}), ("aggregate", {"partition": "final_validation"}),
]
""")
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(runtime.PROJECT_ROOT)}},
    ).execute()
    sources = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "RUN_" not in sources
    assert "INSTALL_DEPENDENCIES" not in sources


def test_review_is_bound_to_preview_and_requires_explicit_input(monkeypatch, tmp_path):
    import IPython.display
    from PIL import Image

    monkeypatch.setattr(IPython.display, "display", lambda *_: None)
    preview = tmp_path / "augmentation_pairs.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(preview)
    matrix_path = tmp_path / "matrix.json"
    runtime.save_json(matrix_path, {"generated_dir": str(tmp_path / "plan")})
    monkeypatch.setattr(
        runtime, "smoke_settings", lambda _: ({"output_dir": str(tmp_path)}, None, None, None)
    )
    monkeypatch.setattr("builtins.input", lambda _: "no")
    with pytest.raises(RuntimeError, match="not recorded"):
        runtime.review_augmentations(matrix_path)
    record = tmp_path / "plan/augmentation_review.json"
    assert not record.exists()
    monkeypatch.setattr("builtins.input", lambda _: "reviewed")
    runtime.review_augmentations(matrix_path)
    assert runtime.load_config(record)["sha256"] == runtime.file_hash(preview)
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("Matching saved review should be reused")
    )
    runtime.review_augmentations(matrix_path)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8)).save(preview)
    monkeypatch.setattr("builtins.input", lambda _: "")
    with pytest.raises(RuntimeError, match="not recorded"):
        runtime.review_augmentations(matrix_path)
