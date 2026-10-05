# Self-Supervised Contrastive Learning for Label-Efficient Medical Image Classification

## 1. Project Context

This project investigates whether self-supervised contrastive learning (SSL) can reduce the amount of labeled data required for clinically useful chest X-ray classification. The dataset is the bundled CheXpert-v1.0-small release. It contains frontal and lateral chest radiographs with multi-label clinical observations, an official training manifest, and an official validation manifest.

The primary comparison is among five image-only self-supervised learning methods: SimCLR, MoCo v2, BYOL, NNCLR, and SwAV. They share the same ResNet encoder, image pool, and augmentation policy. Each encoder is evaluated with a frozen linear probe and full fine-tuning on fixed labeled-patient budgets, alongside supervised-from-scratch and ImageNet-initialized reference baselines.

The project should prioritize a reliable, reproducible comparison over maximizing a single validation score.

## 2. Research Question and Hypotheses

### Primary research question

How do SimCLR, MoCo v2, BYOL, NNCLR, and SwAV pretraining on unlabeled CheXpert radiographs compare in downstream multi-label classification at limited label budgets?

### Hypotheses

1. SSL-pretrained encoders will improve over supervised training from scratch at low labeled-data budgets.
2. Relative performance will differ among the five SSL objectives and may vary with label budget.
3. Fine-tuning and frozen linear probing provide complementary measures of transfer quality.
4. Gains will differ by pathology because label prevalence, uncertainty, and image appearance differ across observations.

## 3. Scope

### In scope

- Multi-label classification of five CheXpert competition observations:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Pleural Effusion
- Five SSL methods (SimCLR, MoCo v2, BYOL, NNCLR, and SwAV) trained on the same training-patient radiographs without labels.
- Label-efficient downstream experiments at fixed label budgets.
- Supervised-from-scratch and ImageNet-initialized reference baselines, linear-probe evaluation, and full fine-tuning.
- Patient-level split controls, reproducible experiment tracking, and clinically cautious error analysis.

### Out of scope for the first version

- Diagnosis or clinical deployment.
- DINO, masked autoencoders, and additional SSL objectives beyond the five defined above.
- Training on external datasets or reporting claims of generalization beyond CheXpert.
- Use of images or labels from the official validation set during model selection.

## 4. Dataset Protocol

### Source files

- `CheXpert-v1.0-small/train.csv`: source of pretraining and downstream-development records.
- `CheXpert-v1.0-small/valid.csv`: final held-out evaluation manifest.
- `CheXpert-v1.0-small/train/` and `CheXpert-v1.0-small/valid/`: image roots referenced by the manifests.

### Record preparation

1. Read the CSV manifests with a structured tabular loader.
2. Convert each manifest path to a local path relative to the dataset root.
3. Keep the source image path, patient ID, study ID, view, and the five selected labels in a prepared manifest.
4. Exclude records whose image path is missing or unreadable, and log the count and paths in a data-quality report.
5. Keep all available image views for pretraining. For downstream experiments, begin with frontal images (`AP` and `PA`) only because they are the most directly comparable; record the filtering count.
6. Convert grayscale images to three identical channels for the shared three-channel backbone interface.

### Target encoding and uncertainty policy

CheXpert labels can be `1` (positive), `0` (negative), `-1` (uncertain), or blank (no mention). The downstream manifest must store both the raw value and the resolved binary target.

Use the CheXpert U-Zeros/U-Ones convention below for the initial study:

| Observation | Positive | Negative or blank | Uncertain (`-1`) |
| --- | --- | --- | --- |
| Atelectasis | 1 | 0 | 1 |
| Cardiomegaly | 1 | 0 | 0 |
| Consolidation | 1 | 0 | 0 |
| Edema | 1 | 0 | 1 |
| Pleural Effusion | 1 | 0 | 0 |

This mapping must be implemented in one reusable preprocessing function and recorded in every run configuration. A later sensitivity experiment may compare this choice with uncertainty masking or alternative mappings, but it must not be mixed into the primary result table.

### Leakage prevention and partitions

- Extract patient identifiers from the image path or manifest metadata before any sampling.
- Partition at patient level, never image level. A patient must appear in exactly one of train, development, or final validation.
- From `train.csv`, make a patient-disjoint development split (target: 10% of patients) for early stopping, threshold selection, and hyperparameter selection. Use iterative multi-label stratification where practical; otherwise document class prevalence before and after splitting.
- Keep `valid.csv` untouched until a configuration has been selected. It is the final held-out evaluation set.
- Create and persist sampled patient IDs for every label budget and seed. Larger budgets should be nested within smaller-budget experiments for each seed where practical.

## 5. Model Design

### Shared encoder

Use ResNet-18 as the default shared encoder for all five SSL methods. ResNet-50
is an optional higher-compute setting. The architecture and input resolution
must be identical across methods in a comparison.

### SSL pretraining

- Pretrain only on images from the training partition; do not use downstream labels.
- Use the method-specific projection/prediction heads and objective for SimCLR, MoCo v2, BYOL, NNCLR, and SwAV. Downstream transfer uses the online encoder only.
- Share the data partition, two-view augmentation policy, optimizer family, and encoder architecture across methods. Method-specific temperatures, queues, momentum, and prototypes are configuration values.
- Use mixed precision when CUDA is available, AdamW, and a cosine learning-rate schedule. The default starting point is batch size 16, 128-pixel input, and 30 epochs; adjust only after recording a short smoke-run throughput and memory profile.
- Save resumable per-epoch checkpoints and the exact configuration.

### X-ray-safe augmentation policy

All five SSL methods use two independently augmented views of each image. X-ray transformations must preserve clinically meaningful anatomy.

- Use random resized crops (scale 0.75–1.0), modest rotation/translation, mild brightness/contrast changes, and occasional 3-pixel Gaussian blur.
- Do not apply hue or saturation changes to grayscale radiographs.
- Avoid aggressive crops, large rotations, posterization, or cutout transforms that can remove pathology-bearing regions.
- Treat horizontal flipping as a configurable ablation. The default primary protocol should disable it because laterality can be clinically informative.

The current torchvision pipeline decodes images from disk on each pass rather than maintaining a full-dataset array cache. This reduces extra disk requirements but makes throughput dependent on storage and DataLoader workers. Training saves example pairs of augmented views before full runs; inspect them for anatomical plausibility.

The implementations use two global views for all five methods. SwAV does not use multi-crop in this study. The MoCo implementation uses a momentum encoder and queue in the single-device workflow; results should be reported with this implementation detail rather than described as a distributed multi-GPU run.

### Downstream models

Train matched downstream protocols with identical data splits, input resolution, label mappings, and evaluation code:

1. **Supervised from scratch**: randomly initialized selected backbone plus classification head, trained only on the chosen labeled subset.
2. **SSL linear probe**: freeze the selected method's pretrained backbone, cache train/development features once, and train only the five-logit classification head.
3. **SSL fine-tuning**: initialize the selected backbone from its method checkpoint, then optimize the encoder and classification head together on the labeled subset.
4. **ImageNet reference**: initialize from the configured torchvision ImageNet weights and fine-tune on the same patient cohort. Record it separately from both scratch and CheXpert SSL initialization.

Use `BCEWithLogitsLoss` with per-label positive weighting computed from the labeled training subset. Store the weights in the run metadata. Optimize AUROC-oriented model selection with the development set; do not select checkpoints from the official validation set.

## 6. Experiment Matrix

### Vision-language model extension

Add a VLM track to determine whether image-text pretraining provides useful medical representations beyond the image-only SSL methods. VLM results must be reported separately because the VLM may have learned from external paired image-report data and therefore answers a different transfer-learning question.

Use CheXzero as the primary VLM baseline because it is a CLIP-style model designed for chest X-ray image-text alignment and naturally supports pathology prompts. Pin the exact checkpoint and implementation revision. If CheXzero cannot be used because of an unavailable checkpoint or incompatible license, use one radiology-domain image-text model with equivalent image-embedding and text-embedding interfaces, and identify it as a protocol deviation in the report. Record its name, revision, source, pretraining data description, license, image preprocessing, and any known CheXpert overlap risk before use. Do not silently substitute a general-purpose natural-image CLIP model for the primary VLM result.

The VLM track has three roles:

1. **Zero-shot prompt classification**: score each X-ray against pathology-present and pathology-absent text prompts without fitting a CheXpert classifier. This measures direct language-grounded transfer.
2. **Frozen VLM feature probe**: freeze the VLM image encoder and train the same five-logit linear head used for image-only SSL. This isolates the quality of its visual representations under the project label budgets.
3. **VLM fine-tuning**: fine-tune the image encoder and classification head at the same label budgets when the checkpoint license and hardware permit. Use a lower encoder learning rate than the newly initialized head.

The primary image-only question compares the five SSL methods with supervised learning from scratch. The VLM extension answers a secondary question about image-text transfer and must be reported separately.

### Prompt protocol for zero-shot VLM evaluation

- Define prompts before evaluating the official validation set and version them in the repository.
- Use paired prompts for every label, for example: `"chest radiograph with pleural effusion"` and `"chest radiograph without pleural effusion"`.
- Use a small, clinically reviewed template set per label rather than selecting a single prompt after observing final-validation results. Templates may vary wording but must preserve the same clinical assertion.
- Average normalized text embeddings across templates for each positive and negative concept.
- Convert image-text similarities into a positive-class score using the positive-versus-negative similarity difference or a two-class softmax. Apply the identical score calculation to every pathology.
- Select prompt templates, optional temperature scaling, and any score-calibration method using only the internal development split. Never tailor prompts to individual final-validation examples.
- Report both uncalibrated zero-shot AUROC/AUPRC and calibrated threshold metrics. Calibration must not alter rank-based AUROC claims.

### VLM analysis protocol

Use VLMs as an analysis tool, not as a source of generated clinical labels or medical conclusions.

- Compare SSL, supervised, and VLM feature spaces using UMAP or t-SNE only as qualitative visualizations; do not treat visual separation as a performance metric.
- Produce class-conditioned retrieval panels: for selected validation queries, retrieve nearest training embeddings and inspect whether anatomy, acquisition artifacts, or pathology cues drive similarity. De-identify and keep examples local to the project.
- Analyze error slices by view position (`AP` versus `PA`), patient age group if available, sex if available, and label prevalence. Omit a slice when its sample size is too small for stable estimates.
- Use Grad-CAM or an equivalent image-attribution method for the classifier head. Mark every heatmap as a post-hoc explanation, not proof of clinical reasoning or localization.
- If a text-generation-capable VLM is used for narrative analysis, restrict it to templated, non-diagnostic descriptions of model outputs and require human review. It must not create ground-truth labels, alter metrics, or be presented as a clinical report generator.
- Maintain an error-analysis worksheet containing run ID, image ID, true labels, predicted probabilities, selected threshold, view, attribution artifact path, and a short reviewer observation.

### Label budgets and seeds

Run each downstream protocol at 1%, 5%, 10%, 25%, and 100% of labeled training patients. Use three fixed seeds for each budget. All methods under a seed/budget pair must receive the exact same persisted sampled patients.

The full image-only matrix produces 180 downstream runs: 12 initialization/protocol arms (five SSL methods each with linear probe and fine-tuning, plus supervised-from-scratch and ImageNet fine-tuning) x 5 budgets x 3 seeds. This matrix is compute-intensive; first validate the pipeline on one method, one budget, and one seed. Each SSL pretraining run is reused across downstream budgets.

### Recommended execution order

1. Verify dataset paths and generate prepared manifests.
2. Produce patient-disjoint partitions and persist their IDs.
3. Run unit tests, check all five SSL losses on a small batch, and complete the bounded end-to-end pretraining smoke.
4. Run a short downstream smoke with one saved patient cohort before launching supervised-from-scratch and ImageNet reference runs.
5. Inspect the shared augmentation pairs, then start full pretraining only after smoke losses are finite.
6. Complete full pretraining for all five methods and archive their checkpoints.
7. Run linear-probe and fine-tuning experiments from smallest to largest label budget.
8. Validate VLM preprocessing and prompt scoring on the development split; then run the zero-shot and frozen-feature experiments.
9. Run VLM fine-tuning only after the frozen-feature and zero-shot results are complete and only at the designated budget tiers.
10. Re-run failed or anomalous jobs only with a recorded reason.
11. Lock selected configurations, evaluate once on official validation data, and generate the final report.

### Initial hyperparameter defaults

These are starting points, not results to tune against the official validation set.

| Component | Initial configuration |
| --- | --- |
| Image size | 128 x 128 default; use the same size across methods |
| SSL encoder | ResNet-18 default; ResNet-50 optional |
| SSL epochs | 30 initial compute-conscious default |
| SSL temperature | 0.1 to 0.2, selected on development protocol |
| SSL batch size | 16 default; increase if GPU memory and throughput allow |
| Downstream epochs | Up to 50 with development-set early stopping |
| Fine-tuning optimizer | AdamW with discriminative or lower encoder learning rate |
| Linear-probe optimizer | AdamW |
| Precision | Automatic mixed precision |
| Random seeds | Three fixed values saved in configuration |

## 7. Metrics and Statistical Reporting

### Primary metric

Macro AUROC across the five observations on the official validation set. This gives every pathology equal influence despite class imbalance.

### Secondary metrics

- AUROC for each observation.
- Macro and per-observation AUPRC.
- Sensitivity, specificity, F1, and balanced accuracy at thresholds selected only on the development set.
- Mean and standard deviation across seeds.
- Patient-level bootstrap 95% confidence intervals for final AUROC and for prespecified SSL-versus-baseline differences.
- Training time, peak GPU memory where available, epochs completed, and checkpoint size.
- Zero-shot VLM metrics with the frozen prompt-set version and calibration status clearly identified.
- Pairwise label-efficiency deltas at each budget: each SSL fine-tuning method versus supervised-from-scratch and ImageNet initialization; compare a VLM frozen probe with the strongest image-only probe selected on development data.
- Slice-level performance with sample counts and confidence intervals where sufficiently powered; do not make subgroup claims from sparse slices.

### Result presentation

- Table: mean +/- standard deviation by method and label budget.
- Plot: macro AUROC versus label fraction, with one curve per method and uncertainty bands.
- Plot: per-pathology AUROC at 1%, 10%, and 100% label budgets.
- Table: final validation metrics and confidence intervals for the selected settings.
- Table: zero-shot, frozen-probe, and fine-tuned results, including a `pretraining modality` column (`none`, `image-only SSL`, `image-text VLM`) to prevent misleading direct claims.
- Qualitative review: selected false positives and false negatives, retrieval examples, and attributions, explicitly framed as model behavior analysis rather than clinical advice.

## 8. Reproducibility and Artifact Layout

Use configuration files rather than hard-coded experiment values. Every run must record:

- Dataset root and a hash or version identifier for prepared manifests.
- Label set, uncertainty mapping, image-view filter, split IDs, label budget, and seed.
- Augmentation parameters, architecture, optimizer, schedule, loss settings, batch size, precision, and software versions.
- Git commit hash, command line, start/end timestamps, hardware, and checkpoint path.
- For every VLM run: checkpoint identifier and revision, model-card URL/source, license review result, image processor version, prompt-set version, embedding normalization, scoring method, calibration method, and trainable modules.

Recommended generated artifact layout:

```text
data/processed/
  manifests/
  splits/
configs/
  pretrain/
  downstream/
  vlm/
  prompts/
outputs/
  pretrain/<run-id>/
  downstream/<run-id>/
  vlm/<run-id>/
  analysis/<run-id>/
  reports/
```

Generated images, checkpoints, cached manifests, and raw experiment outputs should be excluded from version control unless deliberately curated as small examples. Commit code, configurations, split definitions, summaries, and documentation.

## 9. Validation and Tests

Before expensive training, implement and run the following checks:

1. Manifest parsing correctly resolves every sampled local image path.
2. Target conversion matches the uncertainty table for synthetic rows containing `1`, `0`, `-1`, and blanks.
3. No patient appears in more than one partition, label-budget subset, or final-validation overlap.
4. The two-view dataset returns distinct, finite tensors of the expected shape for each SSL method; inspect representative pairs for anatomical plausibility.
5. The model returns five logits per image and the loss remains finite for an imbalanced mini-batch.
6. A checkpoint reload produces the same evaluation outputs for fixed inputs.
7. AUROC and AUPRC functions match known values on small synthetic examples and handle labels with a single class gracefully.
8. A short end-to-end run writes metrics, configuration, split ID, checkpoint, and augmentation samples to the expected run directory.
9. The VLM processor accepts the same sampled image records, emits finite embeddings of stable dimension, and preserves record order.
10. Prompt scoring returns one score per image and pathology, handles template aggregation deterministically, and does not use labels at inference time.
11. Frozen VLM features cannot receive gradients during a linear-probe run; selected modules receive gradients during the fine-tuning configuration.
12. Retrieval and attribution analysis artifacts retain only approved metadata and map back to the exact evaluated checkpoint and run ID.

## 10. Milestones and Acceptance Criteria

### Milestone 1: Data foundation

Prepared manifests, data-quality report, uncertainty mapping, and persisted patient-level splits exist. All data tests pass.

### Milestone 2: Baselines

The supervised-from-scratch and ImageNet-initialized 100% runs complete, produce development metrics, and can be evaluated reproducibly from their checkpoints.

### Milestone 3: SSL representation

All five SSL objectives pass the small-batch smoke tests. The bounded end-to-end pretraining run completes with inspected augmentations, finite loss, a saved checkpoint, and run metadata.

### Milestone 4: Label-efficiency study

All methods complete for all five budgets and three seeds, or every missing run has a documented operational reason. Results are aggregated from raw metric files, not manually copied.

### Milestone 5: VLM comparison and analysis

The zero-shot prompt baseline and frozen VLM probes complete with versioned prompts and model provenance. Fine-tuned VLM results complete for the designated budgets if feasible. Retrieval, attribution, and error-slice analyses are tied to frozen result artifacts.

### Milestone 6: Final analysis

The final report includes the primary AUROC comparison, secondary metrics, confidence intervals, learning curves, limitations, and reproducibility instructions.

## 11. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Patient leakage inflates metrics | Persist and test patient-level splits before any model run. |
| Uncertain labels alter conclusions | Use one declared primary mapping and report a separate sensitivity analysis only if time permits. |
| Class imbalance hides poor minority-label performance | Report both AUROC and AUPRC per label; use weighted loss. |
| SSL augmentations corrupt clinical features | Save and inspect paired views; keep transforms conservative. |
| SSL batch does not fit GPU memory | Use mixed precision and lower the configured batch size; record the setting because it affects method comparison. |
| Repeated validation-set tuning overfits results | Use the internal development split for all selection and evaluate the official validation manifest only after configurations are locked. |
| Small low-budget subsets are unstable | Use fixed multi-seed patient samples and report variance, not just the best run. |
| VLM pretrained on data overlapping CheXpert | Review the model card and pretraining documentation; disclose known or unresolved overlap and label results as transfer benchmarks, not independent generalization. |
| Prompt wording changes zero-shot score | Pre-register a small versioned template set and tune only on the internal development split. |
| VLM is too expensive to fine-tune | Complete zero-shot and frozen-feature probes first; use parameter-efficient fine-tuning only if it is clearly reported as such. |
| Interpretability artifacts are over-read | Present retrievals and attributions as qualitative evidence of model behavior, never as clinical validation. |

## 12. Final Deliverables

1. A documented, configuration-driven training pipeline.
2. Patient-level manifests and reproducible split files.
3. Supervised, SSL linear-probe, and SSL fine-tuned checkpoints and metric logs for all five objectives.
4. Versioned VLM prompt sets, zero-shot scores, frozen-feature probes, and optional VLM fine-tuning artifacts.
5. Aggregated tables and plots for the image-only and image-text label-efficiency comparisons.
6. A final technical report that states the question, methods, results, VLM provenance, limitations, and conditions needed to reproduce the study.

## 13. Implementation Status (2026-10-05)

The repository contains the configured data and training pipeline. The image-only
training pipeline has not yet been run end to end in this workspace; the local
pytest attempt could not collect tests because PyTorch, NumPy, and pandas are not
installed here. No performance claim, checkpoint, split artifact, or
final-validation result is implied by code presence alone.

### Implemented data and reproducibility foundation

- `scripts/prepare_data.py` prepares portable manifests, resolves the declared
  U-Ones/U-Zeros targets, verifies that images exist and can be decoded, and
  emits missing/unreadable-image reports.
- It creates a deterministic patient-level train/development split, writes the
  patient IDs for train, development, and official validation, and records
  target prevalence plus content hashes for every generated manifest.
- It also writes nested patient-ID JSON cohorts for label fractions `0.01`,
  `0.05`, `0.10`, `0.25`, and `1.0`, using seeds `42`, `43`, and `44`. The
  current patient split is deterministic shuffled assignment rather than
  iterative multi-label stratification; prevalence reporting documents the
  resulting distribution.
- For previously prepared data, `scripts/repair_pretrain_manifest.py` creates a
  separate development-patient-free pretraining manifest from the existing
  pretraining and development CSVs. It does not open images or modify cohort
  files; pretraining configs point to this repaired manifest.
- Each training/evaluation output receives `run_metadata.json` with the full
  configuration, command, Git revision, Python/PyTorch versions, CUDA status,
  hardware identifier, and timestamp.

### Implemented image-only experiments

- `scripts/train_pretrain.py` supports SimCLR, MoCo v2, BYOL, NNCLR, and SwAV
  using the shared ResNet-18/ResNet-50 encoder interface. Each method writes
  augmentation examples and resumable `best.pt`/`last.pt` checkpoints that
  include transferable online-encoder weights.
- `scripts/prepare_smoke_manifest.py` creates a bounded unlabeled manifest for
  an end-to-end SimCLR pipeline smoke check. `tests/test_ssl_methods.py` checks
  finite loss, encoder gradients, and update hooks for all five objectives.
- `scripts/train_downstream.py` supports supervised training, SSL frozen linear
  probes, and SSL fine-tuning. Linear probes extract frozen train/development
  features once before fitting the five-label head. Runs accept persisted
  sampled-patient JSON files, use subset-specific positive weights, select
  checkpoints by development macro AUROC, and save development-selected thresholds.
- `scripts/evaluate_checkpoint.py` is the explicit locked-configuration path
  for the official validation manifest. It writes metrics, per-image
  probabilities, threshold metrics, and a patient-resampled macro-AUROC CI.
  It must not be used before configuration selection is complete.

### Implemented VLM extension

- The optional `vlm` dependency group supplies a Hugging Face adapter for
  CLIP-compatible radiology checkpoints. All supplied VLM configs intentionally
  contain required checkpoint, revision, and model-card placeholders.
- `configs/prompts/chexpert_v1.yaml` versions paired positive/negative templates
  for all five observations. `scripts/evaluate_vlm_zeroshot.py` averages
  normalized template embeddings and uses a two-class image-text softmax score
  without labels at inference.
- `scripts/train_vlm.py` supports frozen feature probes and image-encoder
  fine-tuning. Fine-tuning uses a separately configurable, lower encoder
  learning rate. VLM configs record the `image-text VLM` modality and require
  license review before use.

### Implemented reporting and analysis support

- `scripts/aggregate_results.py` produces raw-metric and seed-summary CSVs from
  run artifacts, retaining method and pretraining-modality columns.
- `scripts/analyze_errors.py` creates an editable false-positive/false-negative
  worksheet containing the run, image/patient/study IDs, label, prediction,
  threshold, view, attribution placeholder, and reviewer-observation field.

### Five-method comparison

- Separate YAML configs define the shared default ResNet-18 pretraining setup
  for each SSL method. ResNet-50 remains an optional higher-compute comparison.
- The online encoder checkpoint from each method can be reused for every
  downstream label budget and seed.
- The single-device PyTorch objectives are SimCLR/NT-Xent, MoCo/momentum
  encoder and queue, BYOL/EMA target, NNCLR/nearest-neighbor support queue, and
  SwAV/prototype assignment with Sinkhorn normalization over two global views.
- Pretraining manifests now contain every available view for training patients
  only, keeping development patients out of SSL representation learning.
- Downstream comparisons must point each method to the same persisted patient
  cohort for each budget/seed pair. Do not use the official validation partition
  for selection.

### Deferred implementation and operational requirements

- Gradient accumulation, warmup, LARS, iterative multilabel stratification,
  pretraining peak-memory logging, final paired-difference bootstrap,
  calibration, report plots, retrieval, UMAP/t-SNE, and Grad-CAM are not
  implemented yet.
- Unit tests cover uncertainty conversion, patient overlap, patient budgets,
  NT-Xent, basic multilabel metrics, and forward/backward checks for all five
  SSL objectives. Run the full suite and an end-to-end GPU smoke test in the
  training environment before long jobs.
- Before expensive runs, inspect `augmentation_pairs.png`, review VLM
  provenance/license/overlap risk, and lock configurations on development data
  before final evaluation.
