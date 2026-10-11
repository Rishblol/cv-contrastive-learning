# CheXpert label-efficiency study: scientific and execution protocol

## 1. Purpose and scope

Compare whether SimCLR, MoCo v2, BYOL, NNCLR, and SwAV representations improve
multi-label chest X-ray classification when labels are limited. Compare every
SSL encoder through a frozen linear probe and full fine-tuning, alongside
supervised-from-scratch and explicitly versioned ImageNet references. Reliability,
matched patient cohorts and reproducibility take precedence over a single score.

The five targets are Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural
Effusion. Use the bundled CheXpert-v1.0-small release. Do not claim generalization
to external populations or clinical diagnostic capability. Teacher–student
pseudo-labeling, other SSL objectives and clinical deployment are out of scope.

## 2. One executable source

`chexpert_ssl.ipynb` contains the implementation, an embedded YAML bundle,
and ordinary sequential execution cells. There are no runtime dependencies on `src/`,
`scripts/` or external `configs/` trees. The test-only loader reads setup and definitions before the study section
without executing real-data experiments; no cell tags are required.
Configuration names such as `configs/experiments/notebook_reference.yaml` are
keys inside the bundle, not files to create. External YAML/JSON configurations
remain supported; inheritance is relative to their containing location.

The approved consolidation retains the hardened image-only implementation and
scientific choices. It replaces the original notebook's weaker resume, cache,
cohort and final-evaluation behavior. Historical seed-42 settings remain an
explicit separate profile. The reference matrix uses the notebook-reference
profile described below. Configuration choices are YAML, not training-loop
constants. Every training/evaluation run saves its resolved configuration and
provenance using `save_run_metadata`.

## 3. Data handling and uncertainty

Sources:

- `CheXpert-v1.0-small/train.csv` and `train/`: SSL, downstream training and
  patient-disjoint development.
- `CheXpert-v1.0-small/valid.csv` and `valid/`: final held-out evaluation only.

Preparation preserves source paths, patient/study identifiers including leading
zeros, views, raw labels and resolved binary targets. Missing or unreadable
images are excluded and logged. Partition patients before filtering views: SSL
retains all views of training patients, including lateral-only patients;
downstream/development retain frontal AP/PA records. Network inputs use three
identical grayscale channels.

`resolve_target` implements the primary policy:

| Label | Positive | Negative / blank | Uncertain (-1) |
| --- | --- | --- | --- |
| Atelectasis | 1 | 0 | 1 |
| Cardiomegaly | 1 | 0 | 0 |
| Consolidation | 1 | 0 | 0 |
| Edema | 1 | 0 | 1 |
| Pleural Effusion | 1 | 0 | 0 |

Alternative uncertainty handling must be declared as a separate sensitivity
experiment with its own configuration and results.

The reference split uses NumPy `RandomState`, seed 1234, with 10% development
patients. Budget seeds are 0/1/2. A persisted permutation produces nested
1/5/10/25/100% patient cohorts with ceiling-rounded counts. All methods at the
same seed/budget reuse the same JSON cohort. Do not resample images or regenerate
cohorts during a comparison. Iterative multilabel stratification is not
implemented; train/development prevalence is recorded.

Legacy patient IDs with lost padding are normalized only when numeric identity
agrees with source paths. Duplicate aliases, unknown cohort members, genuine ID
mismatches and partition overlap fail validation. Corrections are recorded;
existing cohort files are not rewritten. The legacy repair task excludes
development patients from an older SSL manifest without resampling cohorts.

Official validation may be read during preparation for manifest creation,
image-quality checks and disjointness verification. Preparation does not report
its label prevalence. It must never inform model/prompt selection, calibration
or hyperparameter tuning. The old notebook's early validation-prevalence output
has been removed. Full final prediction/metrics require development-selection
locks.

## 4. Directories and cache

Paths are relative to the repository root, discovered from the notebook or
explicitly set with `CHEXPERT_PROJECT_ROOT`. The staged API sets and restores the
working directory for each operation. No automatic download or credential-copy
step is performed. Use local Jupyter Notebook, JupyterLab or VS Code.
The first cell sets the kernel working directory to the notebook directory.
A dependency-install cell uses the active kernel Python if required imports
are missing. Direct managed training/evaluation calls re-enter
the project root even if the user subsequently changes the working directory.

Reference artifacts:

```text
data/processed/notebook-reference/
  manifests/  all_training.csv, pretrain_train_leakage_free.csv,
              downstream_train.csv, development.csv, final_validation.csv
  splits/     patient partitions and budget_<fraction>_seed_<seed>.json
  cache/      all_training_gray_128.npy plus cache signature
  smoke/      bounded, versioned smoke inputs
outputs/notebook-consolidated/
  pretrain/ downstream/ evaluation/ smoke/ reports/
outputs/plans/notebook-consolidated/
```

The `pretrain_train.csv` compatibility alias is retained for prepared-data
compatibility. Existing data/results/plans are not deleted by consolidation.

Cache grayscale uint8 images with PIL bilinear resizing and a manifest/resolution
signature; decode in bounded threaded batches. SSL indexes training patients
only, even though the shared training cache also contains development images
for downstream evaluation. Development images do not enter SSL learning.

Notebook data loaders require `num_workers: 0`: definitions in Jupyter's kernel
are not portable importable worker modules. Threaded decoding and batched
device-side augmentation remain available. This operational change may affect
throughput and does not promise equivalence to prior GPU trajectories.

## 5. Encoders, augmentations and pretraining

The primary shared encoder is ResNet-18 at 128 pixels; ResNet-50 remains an
optional separate comparison with independent output/plan paths. Additional
encoder interfaces retained in the implementation are not part of the primary
matched study. Compare all five methods with identical backbone, image pool,
resolution, augmentation and shared optimizer settings.

Reference defaults:

- 30 epochs, batch size 256, AdamW weight decay 0.0001.
- Base LR `0.001 * batch_size / 256`, two warmup epochs, per-step cosine decay
  with minimum multiplier 0.001.
- Projection/prediction hidden width 1024, embedding dimension 128.
- Grayscale normalization mean 0.449, standard deviation 0.226.
- Conservative two-view crop scale 0.6–1.0, ratio 0.85–1.18; rotation 10 degrees,
  translation 0.05, brightness 0.1, contrast 0.2; occasional blur and mild noise.
  Horizontal flip is disabled. Inspect saved augmentation pairs before full runs.
- AMP, channels-last layout, TF32, cuDNN benchmarking and fused AdamW are
  configurable CUDA performance options, not guarantees of bitwise determinism.

SimCLR uses NT-Xent; MoCo v2 uses a momentum encoder, queue and four-way split
BatchNorm; BYOL updates the EMA target momentum per step; NNCLR uses detached
nearest-neighbor positives as InfoNCE anchors; SwAV uses two global views with
log-space Sinkhorn assignments and freezes prototypes for the first epoch.
SwAV multi-crop and distributed multi-GPU training are not implemented.

Monitor feature spread each epoch for collapse. Nonfinite losses fail promptly.
`best.pt` minimizes SSL training loss; reference transfer uses the final online
encoder from `last.pt`. Pretraining seed 0 is reused across downstream seeds;
downstream variability does not estimate pretraining-seed variability.

## 6. Downstream matrix

Five SSL methods × two transfer protocols, plus scratch and ImageNet, yield
12 arms × five patient budgets × three seeds = 180 downstream runs. Within
ascending budget and seed, run scratch, ImageNet, all probes, then all fine-tunes.
A subset may be selected explicitly using stage filters.

- Probes freeze/evaluate the encoder, standardize features with training-only
  mean and sample standard deviation (floor 1e-6), and save statistics in the
  head. Batch size 512, head LR 0.001, maximum 100 epochs, patience 10.
- Other arms use batch size 64, patience 7, and budget-specific epoch caps
  30/30/25/20/10. Head LR 0.001; pretrained encoder LR 0.0001, scratch 0.001.
  Fine-tuning uses cosine decay.
- Downstream device augmentation: crop scale 0.8–1.0, ratio 0.9–1.1, rotation
  5 degrees, translation 0.03, brightness/contrast 0.1; no blur, noise or flip.
- BCE positive weights come only from the selected training subset, clipped
  to [1, 50]. Development macro AUROC selects checkpoints; development Youden
  thresholds are embedded in the selected checkpoint.
- ImageNet weights are explicitly versioned and reported as a separate
  initialization, not CheXpert image-only SSL.

## 7. Checkpoints, provenance and evaluation

Training uses atomic full-precision best/last checkpoints containing model,
optimizer, scheduler/scaler where applicable, training signature and RNG state.
OS-backed locks reject concurrent writers to a run. Failure/interruption status
is recorded. Resume validates resolved settings, input hashes and saved notebook
implementation hashes. CPU tests check interrupted versus uninterrupted weights.

All saved executable cells contribute to the implementation digest; notebook
outputs, markdown and the YAML configuration cell do not. Resolved settings enter the run
signature independently. Save code edits, restart and rerun definitions before
training. Editing saved implementation while a kernel is loaded is rejected by
the digest check. Unsaved editor changes must not be treated as recorded code.
Use new output/plan directories when code or experimental settings change.
The consolidated reference profile starts under fresh
`outputs/notebook-consolidated/` and `outputs/plans/notebook-consolidated/` roots,
reusing the existing reference data and cohorts. Old output directories remain
intact. Old package and original standalone-notebook checkpoints are not automatic resume points.
Compatible encoder state dictionaries may be transfer inputs after validation.

`lock_run` records the development-only selection reason and binds the
checkpoint, training signature, development metrics and thresholds by hashes.
Locked training runs are immutable. No automatic winner selection occurs.
Evaluation verifies every requested lock before opening held-out records,
derives preprocessing and architecture from the checkpoint, rejects conflicting
overrides, checks patient disjointness and records lineage. Evaluation never
downloads initialization weights or fits thresholds. Existing completed final
results are reused only after validation; direct repeat evaluation is rejected.

Report macro and per-label AUROC/AUPRC, sensitivity, specificity, F1 and balanced
accuracy at development thresholds. Final evaluation exports predictions, patient
IDs, AP/PA and available age/sex, plus patient-bootstrap macro and per-label
AUROC confidence intervals. Small held-out sample sizes can produce wide intervals.

Aggregation reads metric artifacts, separates development/final-validation
partitions, excludes smoke/archive results and rejects duplicate paired
seed/method/budget records. Export raw metrics, mean/std/count summaries,
label-efficiency tables and plots. Never manually transcribe study scores.
Paired-difference bootstrap and probability calibration remain future work.

## 8. Verification and execution gates

The sequence is prepare → tests/bounded smoke → human augmentation review →
pretrain → downstream → development report → selection locks → final evaluation
→ final report. Run All executes each study stage directly, including the declared full
downstream matrix. Human input is requested after displaying augmentation pairs
and before locking configurations based on development results. Matching saved
reviews and verified selection locks are reused. There are no cell tags or
per-stage enable switches.

`run_stage("smoke")` runs pytest, all five SSL objectives for two epochs with at
most four steps/epoch at batch size 64, and scratch/SimCLR probe/MoCo fine-tune/
ImageNet downstream paths from bounded saved cohorts. ImageNet smoke requires
its pinned weights. Smoke data/results are versioned by inputs, implementation
and settings. Pretraining requires a matching test-completion record, completed
smoke metadata and an explicitly reviewed augmentation image hash.

Regression tests load definition cells only and cover targets, identifiers,
cohorts, losses, both ResNet interfaces, numerical stability, checkpoint transfer,
CPU interruption/resume, locks, evaluation and aggregation. Run notebook schema,
lint/format and Run All sequencing checks as part of consolidation verification.
Synthetic CPU results are engineering checks, not CheXpert performance evidence.
The full reference GPU smoke and human image review must run on the intended
training environment before expensive jobs. Failed/anomalous reruns need a
recorded operational reason.

## 9. Optional image-text VLM extension

VLM adapters and development tasks are retained as disabled, experimental
scaffolding. Install `.[vlm]`. Require an explicitly reviewed radiology-domain
checkpoint, immutable 40-character revision, HTTPS model-card source, license
review, overlap review and `pretraining_modality: image-text VLM`. Placeholder
configs fail before any weight download; no general-purpose CLIP fallback.
CheXzero is the intended baseline, subject to adapter compatibility and review.

Prompts are versioned positive/negative templates per pathology. Average
normalized template embeddings and use a two-class similarity softmax. Select
prompts/calibration on development only. Probes/fine-tuning must use the same
persisted patient cohorts and target mapping as image-only arms. Frozen VLM
encoders stay in evaluation mode. These tasks reject held-out manifests and
existing run directories; full image-only resume/selection-lock behavior has
not been extended to VLM final evaluation.

Error worksheets retain run/image/patient/study identifiers, target, probability,
threshold, view and reviewer observations. Retrieval, UMAP/t-SNE, attribution,
paired bootstrap and narrative VLM analysis are future work. They must not create
ground-truth labels, supply diagnoses or imply clinical validity.

## 10. Local Jupyter simplification

The executable is now named `chexpert_ssl.ipynb`; cloud-drive setup and
cloud-specific instructions are removed. Project root is the directory containing
this notebook. Root setup uses only the Python standard library and runs before
conditional dependency installation and third-party imports. Experiment settings,
cohort paths and artifact locations are unchanged.

SimCLR and the public loss tests now use one `nt_xent_loss` implementation, with
the same float32 normalization and objective. The unused ad-hoc patient-budget
sampler is removed; selection uses persisted cohorts. Redundant loss and generic
boilerplate blocks are removed. Historical profiles and optional reviewed VLM
functions remain because they serve distinct purposes.

## 11. Sequential Run All execution

The user requested an ordinary executable notebook. All library/workflow tags
and `RUN_*` switches are removed. Setup installs missing dependencies, and the
study cells call preparation/reuse, smoke, pretraining, the complete configured
downstream matrix, development reporting, selection, final evaluation and final
reporting in order. Existing preparation is checked against the requested split
settings, required manifests, patient disjointness and every persisted cohort.
It is reused without rewriting cohorts.

A displayed augmentation preview requires the user's `reviewed` input unless
its exact saved hash has already been reviewed. Completed development results
require a user-supplied selection reason before all declared arms are locked.
Neither decision is inferred from held-out data or automatically approved.
Tests never run the real-data study: the loader stops before its execution
section, sequencing tests substitute tasks, and kernel integration uses synthetic
data. This changes notebook operation, not the scientific comparison.
