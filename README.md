# Self-Supervised CheXpert Classification

This repository implements the image-only core of a label-efficient chest X-ray
classification study: SimCLR pretraining, supervised baselines, frozen linear
probing, and SimCLR fine-tuning. The full methodology, including the planned
CheXzero VLM extension, is in [docs/context.md](docs/context.md).

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Use Python 3.10-3.13. The active local Python 3.14 environment does not currently
have the required PyTorch package installed.

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

Each downstream configuration uses the same patient-level label fraction and seed.
Change `label_fraction`, `seed`, and `output_dir` to run the full experiment matrix.

## Verification

```powershell
pytest
```

The current implementation covers the reproducible image-only core. The next
implementation stage adds CheXzero checkpoint loading, versioned prompt scoring,
and VLM feature probes without changing the dataset or metric interfaces.
