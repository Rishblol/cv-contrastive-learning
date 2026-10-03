# Self-Supervised CheXpert Classification

This repository implements the full, configuration-driven study protocol in
[docs/context.md](docs/context.md): patient-safe CheXpert preparation, SimCLR
pretraining, supervised/linear-probe/fine-tuning comparisons, locked final
evaluation, and an optional CheXzero-compatible VLM track.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Use Python 3.10-3.13. The active local Python 3.14 environment does not currently
have the required PyTorch package installed.

## Model comparison

The shared image-only backbone set is ResNet-18, ResNet-50, DenseNet-121,
EfficientNet-B0, and ViT-B/16. Every backbone uses the same input interface and
five-label head. Experiment choices are listed in
`configs/models/backbones.yaml`; run-specific resolved configurations are saved
with each output. ViT-B/16 expects the shared 224 x 224 input size.

## Prepare manifests

This reads the bundled dataset, applies the declared CheXpert uncertainty policy,
filters invalid paths, and creates patient-disjoint train/development partitions.

```powershell
python scripts/prepare_data.py --dataset-root CheXpert-v1.0-small --frontal-only-downstream
```

The official validation manifest is written to `data/processed/manifests/final_validation.csv`.
Do not use it for model or prompt selection.

## Train the image-only study

```powershell
python scripts/train_pretrain.py --config configs/pretrain/simclr.yaml
python scripts/train_downstream.py --config configs/downstream/supervised.yaml
python scripts/train_downstream.py --config configs/downstream/simclr_linear.yaml
python scripts/train_downstream.py --config configs/downstream/simclr_finetune.yaml
```

`prepare_data.py` persists nested patient cohorts for every budget and seed in
`data/processed/splits/`. Reference the appropriate JSON through
`sampled_patients` in each downstream/VLM config so every method uses identical
patients. The initial fixed seeds are 42, 43, and 44; budgets are 1%, 5%, 10%,
25%, and 100%.

## Locked final evaluation and reporting

After selecting a configuration exclusively on the development split, point
`configs/evaluation/final_validation.yaml` at its checkpoint and thresholds:

```powershell
python scripts/evaluate_checkpoint.py --config configs/evaluation/final_validation.yaml
python scripts/aggregate_results.py
python scripts/analyze_errors.py --predictions outputs/reports/.../predictions.csv --thresholds outputs/downstream/.../thresholds.json --run-id RUN_ID --output outputs/analysis/RUN_ID/errors.csv
```

This is the only command path intended to read the official validation manifest.

## Optional VLM extension

Install the optional dependencies, review and pin a CheXzero-compatible model
checkpoint/revision/license, then replace the explicit placeholders in
`configs/vlm/`. Prompts are versioned in `configs/prompts/chexpert_v1.yaml`.

```powershell
pip install -e ".[vlm]"
python scripts/evaluate_vlm_zeroshot.py --config configs/vlm/zeroshot.yaml
python scripts/train_vlm.py --config configs/vlm/linear_probe.yaml
python scripts/train_vlm.py --config configs/vlm/finetune.yaml
```

VLM results are always marked `image-text VLM` and must be reported separately
from the image-only SimCLR comparison.

## Verification

```powershell
pytest
```

The implementation includes the image-only core and optional VLM tooling. VLM
configs contain placeholders and require checkpoint, revision, source, and
license review before use. Run focused tests before a full training run, inspect
the saved SimCLR augmentation pairs, and select settings on the development
partition before locked final evaluation.
