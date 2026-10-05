# Self-Supervised CheXpert Classification

This repository compares five image-only self-supervised learning (SSL)
methods using a shared ResNet-18 encoder: SimCLR, MoCo v2, BYOL, NNCLR, and
SwAV. Each learned encoder is evaluated with a frozen linear probe and full
fine-tuning across the same patient-level label budgets. A supervised-from-
scratch model is the reference baseline. ResNet-50 can be selected in the
pretraining and downstream YAML files when additional compute is available.

The workflow is implemented as Python scripts and YAML configurations. The
notebook in `notebooks/` is a reference implementation, not a required runtime.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## Prepare data

```bash
python scripts/prepare_data.py --dataset-root CheXpert-v1.0-small --frontal-only-downstream
```

The script writes manifests, patient cohorts, and provenance to `data/processed/`.
Pretraining uses every view from training patients only; development patients
are excluded from both SSL and downstream training. The official validation
partition must not be used for model selection.

If these manifests were generated before this pipeline update, repair the
existing pretraining manifest instead of rerunning the image scan:

```bash
python scripts/repair_pretrain_manifest.py
```

This writes `pretrain_train_leakage_free.csv` and a repair report. It leaves the
source manifest and saved patient cohorts unchanged. All five pretraining YAML
configs and the smoke-manifest helper use the repaired manifest by default.

If the manifests and cohorts are already prepared, continue with tests and the
short smoke run. The smoke config uses a small ResNet-18 and one epoch:

```bash
pytest
python scripts/prepare_smoke_manifest.py
python scripts/train_pretrain.py --config configs/pretrain/simclr_smoke.yaml
```

Inspect `outputs/smoke/simclr-resnet18/augmentation_pairs.png` before full
training.

## Pretrain the five methods

Each method has its own resumable config and checkpoint directory. Defaults use
ResNet-18, 128-pixel images, batch size 16, and 30 epochs to fit a modest GPU.
Run them individually; completed epochs are saved in `last.pt`.

```bash
python scripts/train_pretrain.py --config configs/pretrain/simclr.yaml
python scripts/train_pretrain.py --config configs/pretrain/moco.yaml
python scripts/train_pretrain.py --config configs/pretrain/byol.yaml
python scripts/train_pretrain.py --config configs/pretrain/nnclr.yaml
python scripts/train_pretrain.py --config configs/pretrain/swav.yaml
```

All five methods use the same PIL/torchvision two-view data pipeline and
ResNet-18 input settings. Images are decoded from disk during training rather
than stored as a full-dataset cache, which avoids a large extra cache but makes
storage throughput and `num_workers` affect epoch time. `train_pretrain.py`
prints periodic step loss and per-epoch duration; use those measurements to
adjust the YAML batch size/epoch count before committing to the full schedule.

## Downstream comparisons

Set `ssl_checkpoint` in `configs/downstream/simclr_linear.yaml` or
`simclr_finetune.yaml` to the selected method's `best.pt` or `last.pt`, and set
`ssl_method` to its name (`simclr`, `moco`, `byol`, `nnclr`, or `swav`). Set
`output_dir` to a unique method/budget/seed path. Use the same persisted
`sampled_patients` JSON for every method at a given budget and seed. For
supervised-from-scratch, use `configs/downstream/supervised.yaml`. To run the
ImageNet reference supported by the same script, set
`imagenet_pretrained: true` in a separate downstream config/output directory;
keep it distinct from the supervised-from-scratch run. Linear probes cache
frozen features once, then train only the small classification head.
Fine-tuning and supervised training select checkpoints on the internal
development partition.

After locking the configuration on development data, run
`scripts/evaluate_checkpoint.py` against the official validation manifest and
aggregate machine-readable metrics with `scripts/aggregate_results.py`.

## Optional VLM extension

VLM experiments are a separate optional track. Install `pip install -e ".[vlm]"`
and review the pinned checkpoint, revision, source, and license requirements in
`AGENTS.md` and `configs/vlm/` before use. Do not silently substitute a general
CLIP model.
