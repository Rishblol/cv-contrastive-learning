# Repository Guidance

## Purpose

This repository implements the CheXpert label-efficiency study described in
`docs/context.md`. Treat that document as the scientific protocol and update it
when changes alter data handling, evaluation, models, or reproducibility.

## Safety and experimental integrity

- Do not use `CheXpert-v1.0-small/valid.csv` or
  `data/processed/manifests/final_validation.csv` for model selection, prompt
  selection, calibration fitting, or hyperparameter tuning.
- Splits and label budgets are patient-level. Reuse the JSON cohorts generated
  in `data/processed/splits/` across all methods for an equal seed/budget pair;
  do not resample images independently.
- Keep the declared U-Ones/U-Zeros mapping in `chexpert_ssl.data.resolve_target`
  as the primary protocol unless the context document and run configuration
  explicitly introduce a separate sensitivity experiment.
- Do not present model output, attribution, retrieval, or VLM text as diagnosis
  or clinical advice.

## Implementation conventions

- Keep experiment choices in YAML configurations rather than hard-coding them.
- Write generated data under `data/processed/` and results under `outputs/`;
  do not commit checkpoints, cached manifests, large images, or raw outputs.
- Every training or evaluation entry point must save its fully resolved config
  and provenance with `save_run_metadata`.
- Preserve the interchangeable ResNet encoder interface (`resnet18` and
  `resnet50`) so smoke configurations can remain lightweight.
- VLM work is optional. It requires the `[vlm]` dependency group, a pinned
  radiology-domain checkpoint/revision, model-card source, license review, and
  explicit `image-text VLM` reporting. Never silently replace it with a
  general-purpose CLIP checkpoint.

## Verification expectations

- Add or update focused tests for changed data, loss, metrics, checkpoint, and
  VLM behavior. Do not claim tests passed unless they were actually run.
- Before expensive training, run the documented test/smoke sequence and inspect
  the generated SimCLR augmentation pairs.
- Use `scripts/evaluate_checkpoint.py` only after a configuration is locked on
  the development partition. Aggregate results from metric artifacts with
  `scripts/aggregate_results.py`, never by manual transcription.
