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
  in `data/processed/**/splits/` across all methods for an equal seed/budget pair;
  do not resample images independently.
- Keep the declared U-Ones/U-Zeros mapping in the notebook’s `resolve_target`
  as the primary protocol unless the context document and run configuration
  explicitly introduce a separate sensitivity experiment.
- Do not present model output, attribution, retrieval, or VLM text as diagnosis
  or clinical advice.

## Implementation conventions

- Keep experiment choices in the notebook’s `CONFIG_YAML` bundle or explicit
  external YAML overrides rather than hard-coding them.
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
- Use the notebook’s `evaluate` / `run_stage("evaluate")` only after a
  configuration is locked on the development partition. Aggregate results from
  metric artifacts with `run_stage("aggregate", partition=...)`, never by manual
  transcription.
- Keep executable implementation in `chexpert_ssl.ipynb`; tests load setup and
  definitions through `tests/notebook_runtime.py` before the study section. Do not reintroduce
  duplicate modules or generated script trees. Use ordinary untagged cells;
  Run All executes the study without enable switches. Keep augmentation review
  and development-only selection locks before their dependent stages.
- Run `python -m pytest -q`, `python -m ruff check .`, and
  `python -m ruff format --check .` after changes. Notebook loaders require
  `num_workers: 0`; cache decoding may use threads.

- Target local Jupyter, with the notebook directory as project root. Keep root
  setup before third-party imports, dependency installation conditional on missing imports, and relative
  data/output paths anchored to the root. Do not add cloud-drive setup.
