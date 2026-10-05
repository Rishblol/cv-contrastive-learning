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

Inspect `outputs/smoke/simclr-resnet18-gpu-cache/augmentation_pairs.png` before full
training.

## Pretrain the five methods

Each method has its own resumable config and checkpoint directory. Defaults use
ResNet-18, 128-pixel images, batch size 64, and 30 epochs. Run them individually;
completed epochs are saved in `last.pt`.

```bash
python scripts/train_pretrain.py --config configs/pretrain/simclr.yaml
python scripts/train_pretrain.py --config configs/pretrain/moco.yaml
python scripts/train_pretrain.py --config configs/pretrain/byol.yaml
python scripts/train_pretrain.py --config configs/pretrain/nnclr.yaml
python scripts/train_pretrain.py --config configs/pretrain/swav.yaml
```

All methods share the same preprocessed cache and GPU-side augmentation
settings. The first run builds a reusable 128 x 128 grayscale uint8 cache at
`data/processed/cache/pretrain_gray_128.npy` (about 3.1 GiB for the current
manifest); the other methods reuse it. Keep this cache on fast pod storage. The
training loop uses pinned-memory batches, prefetch workers, channels-last
convolutions, mixed precision, TF32, and fused AdamW when CUDA is active. Logs
show the selected GPU, steps/second, allocated GPU memory, epoch duration, and
images/second. If batch size 64 does not fit, lower it in all five configs to
the same value (start with 32).

The optimized configs use separate `*-gpu-cache-*` output directories, so the
old one-epoch SimCLR checkpoint will not be resumed with the new augmentation
and batching settings. The first full config run includes cache construction;
subsequent pretraining methods reuse the completed cache.

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
