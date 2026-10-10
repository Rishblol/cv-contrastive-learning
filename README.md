# Self-Supervised CheXpert Classification

This repository compares **SimCLR, MoCo v2, BYOL, NNCLR, and SwAV** using a shared
ResNet encoder, training image pool, augmentation policy, and patient cohorts.
Each SSL encoder supports frozen linear probing and full fine-tuning. Scratch
and explicitly versioned ImageNet initialization provide reference baselines.
The scientific protocol is [docs/context.md](docs/context.md).

## Setup

Run commands from the repository root with Python 3.10–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.
A CUDA-capable training environment is recommended for the complete study.

## Configuration and execution order

`configs/common.yaml` inherits the ResNet-18/128 defaults from
`configs/models/backbones.yaml`. `configs/pretrain/defaults.yaml` defines the
shared SSL training and augmentation settings; the five method files define
only their objectives and output paths. `configs/downstream/defaults.yaml`
defines matched downstream settings. YAML `extends` paths are relative to the
containing YAML; dataset and output paths are relative to the repository root.
Resolved configurations are saved beside every run.

`configs/experiments/label_efficiency.yaml` declares five pretraining runs and
180 downstream runs: ten SSL transfer arms, scratch, and ImageNet, across five
patient budgets and three seeds. Pretraining seed 42 is reused across downstream
budgets/seeds; downstream variability does not measure pretraining-seed variance.
The planner validates settings across all arms and writes resolved configurations
and commands under `outputs/plans/label-efficiency/`. Planning starts no training:

```bash
python scripts/run_experiments.py --stage plan
```

Generated plans are immutable: changing an existing plan requires a new
`generated_dir`. Use new output directories when changing experimental settings.
For a ResNet-50 comparison, change the shared backbone setting and give the
pretraining, downstream, cache, and plan artifacts separate paths.

### 1. Prepare data once

Set the dataset location in `configs/data/prepare.yaml`, then run:

```bash
python scripts/run_experiments.py --stage prepare
```

Preparation partitions all training patients before filtering downstream records
to frontal views. SSL retains every available view of training patients,
including patients with only lateral images. It writes the canonical
`pretrain_train_leakage_free.csv`, the compatibility alias `pretrain_train.csv`,
downstream/development/final manifests, quality reports, partition IDs, and
nested patient-budget JSON files. Patient identifiers retain leading zeros.
Existing prepared data is protected against accidental regeneration.

Reuse existing manifests/cohorts if already prepared. For older pretraining
manifests, the migration command is:

```bash
python scripts/repair_pretrain_manifest.py
```

It creates a separate development-patient-free manifest without scanning images
or modifying saved cohorts. Explicit `prepare_data.py --overwrite` regenerates
the prepared study; do not use it to change cohorts within an existing comparison.
Official validation records are never used for training or development selection.

### 2. Run tests and bounded smoke checks

```bash
python scripts/run_experiments.py --stage smoke
```

This runs the test suite, creates a bounded SSL manifest, trains a one-epoch
SimCLR smoke, and creates a bounded downstream smoke from the persisted 1%
cohort. The downstream smoke uses separate manifests/cohort/output paths.
Inspect `outputs/smoke/simclr-resnet18-gpu-cache/augmentation_pairs.png` for
anatomical plausibility. Smoke configs use `resume: false`; to repeat a completed
smoke, choose new smoke output paths in the smoke YAMLs.

### 3. Pretrain all five methods

After inspecting the augmentation preview:

```bash
python scripts/run_experiments.py --stage pretrain --augmentation-reviewed
```

The runner requires completed pretraining and downstream smoke artifacts and
records the reviewed preview hash. Defaults are 30 epochs, batch size 64,
128-pixel images, AdamW, and a cosine schedule. A shared grayscale memory-mapped
cache reduces repeated decoding. CUDA runs support automatic mixed precision,
channels-last layout, TF32, and fused AdamW. If memory is insufficient, lower
batch size in the shared pretraining defaults for all five methods and use new
run/plan directories. Individual entry points remain available, for example:

```bash
python scripts/train_pretrain.py --config configs/pretrain/moco.yaml
```

Pretraining `best.pt` minimizes the SSL training loss; `last.pt` is resumable.
Neither is selected using official validation. MoCo is single-device with a
momentum encoder/queue; SwAV uses two global views without multi-crop.

### 4. Run matched downstream experiments

Start with one budget and seed:

```bash
python scripts/run_experiments.py --stage downstream --methods simclr scratch imagenet --budgets 0.01 --seeds 42
```

Then execute the complete downstream matrix:

```bash
python scripts/run_experiments.py --stage downstream
```

Every method at a seed/budget pair consumes the same persisted JSON cohort.
Linear probes extract frozen features once per run. Fine-tuning supports a
separate encoder learning rate. All runs use subset-specific positive weights,
development macro AUROC for checkpoint selection, and development-selected
thresholds. `best.pt` embeds those thresholds.

Checkpoint writes are atomic; concurrent writers to one run are rejected.
Resume verifies the resolved configuration, implementation and input hashes and restores RNG
state. CPU interruption tests check identical resumed weights. CUDA defaults
favor throughput and do not promise bitwise determinism. Older checkpoints
without the current signature/RNG state remain transfer inputs if compatible,
but require a new output directory for training.
Completed jobs are reused only when their configurations, input hashes, and
implementation and required artifacts still match. Failed jobs stop the runner and can be resumed
with the same configuration. Record the operational reason before rerunning an
anomalous experiment.

### 5. Review development results and lock selection

Development summaries can be generated separately:

```bash
python scripts/aggregate_results.py --partition development
```

After completing development-only configuration selection, lock each selected
run with a reason describing that decision:

```bash
python scripts/lock_selection.py --run-dir outputs/downstream/simclr-ssl_finetune-resnet18-budget0p01-seed42 --reason "Configuration selected from development results"
```

Locks bind the training signature, checkpoint, thresholds, and development
metrics. Locked training runs are immutable. The runner does not select or lock
models automatically.

### 6. Evaluate and aggregate held-out artifacts

```bash
python scripts/run_experiments.py --stage evaluate
python scripts/run_experiments.py --stage aggregate
```

Use `--methods`, `--budgets`, and `--seeds` to evaluate a selected subset. Every
requested lock is checked before opening held-out records. Evaluation derives
architecture, input resolution, and preprocessing from the checkpoint and
rejects conflicting overrides. Existing final results are reused by the runner;
the evaluation entry point rejects accidental repeat evaluation.

Final artifacts live under `outputs/evaluation/`; summaries live under
`outputs/reports/final_validation/`. Development summaries are separate.
Aggregation never mixes development and held-out scores and rejects duplicate
seed/method/budget results. It includes per-label and threshold metrics in
addition to macro metrics. Evaluation exports predictions, AP/PA and available
age/sex metadata, plus a patient-bootstrap macro-AUROC confidence interval.
Paired-difference bootstrap, report plots, retrieval, and attribution remain
future reporting work.

## Optional VLM extension

VLM experiments are a separate image-text transfer track outside the image-only
runner. Install `pip install -e ".[vlm]"` and review the pinned radiology checkpoint,
revision, model-card source, license, and overlap risk before using `configs/vlm/`.
Those configurations intentionally retain placeholders. The VLM adapter and
lifecycle have not received the image-only production hardening described above.
Never silently substitute a general-purpose CLIP checkpoint.
