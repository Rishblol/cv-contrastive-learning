# Self-Supervised Contrastive Learning for Label-Efficient Medical Image Classification

## 1. Project Context

This project investigates whether self-supervised contrastive learning (SSL) can reduce the amount of labeled data required for clinically useful chest X-ray classification. The dataset is the bundled CheXpert-v1.0-small release. It contains frontal and lateral chest radiographs with multi-label clinical observations, an official training manifest, and an official validation manifest.

The main method is SimCLR: an encoder learns image representations from different augmented views of the same X-ray without using disease labels. The pretrained encoder is then evaluated on downstream disease prediction using progressively smaller labeled subsets. Its results are compared to models trained with labels alone.

The project should prioritize a reliable, reproducible comparison over maximizing a single validation score.

## 2. Research Question and Hypotheses

### Primary research question

Does SimCLR pretraining on unlabeled CheXpert radiographs improve downstream multi-label classification performance when only a small fraction of labels is available?

### Hypotheses

1. A SimCLR-pretrained encoder will outperform a supervised model trained from scratch at 1%, 5%, 10%, and 25% labeled-data budgets.
2. The performance advantage will be largest at the lowest label budgets and narrow as the budget approaches 100%.
3. Fine-tuning the full pretrained encoder will generally outperform a frozen linear probe, while the linear probe will show whether the learned representations are directly useful.
4. Gains will differ by pathology because label prevalence, uncertainty, and image appearance differ across observations.

## 3. Scope

### In scope

- Multi-label classification of five CheXpert competition observations:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Pleural Effusion
- SimCLR pretraining on all available training radiographs without labels.
- Label-efficient downstream experiments at fixed label budgets.
- Supervised baselines, linear-probe evaluation, and full fine-tuning.
- Patient-level split controls, reproducible experiment tracking, and clinically cautious error analysis.

### Out of scope for the first version

- Diagnosis or clinical deployment.
- Comparison with multiple SSL families such as MoCo, BYOL, DINO, or masked autoencoders.
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

Compare five interchangeable backbones: ResNet-18, ResNet-50, DenseNet-121,
EfficientNet-B0, and ViT-B/16. All receive normalized three-channel X-rays and
return one feature vector per image; a shared linear head produces five logits.
Convolutional backbones use their pooled feature representation. ViT-B/16 uses
its class-token representation. All are randomly initialized for the primary
image-only comparison; ImageNet initialization is a separate configurable
experiment and must be reported distinctly.

Use the same architecture set across the supervised-from-scratch, SimCLR linear
probe, and SimCLR fine-tuning methods. Image resolution and architecture are
configuration values, and each run records its backbone so metrics can be
compared by architecture and method.

### SimCLR pretraining

- Pretrain only on images from the training partition; do not use downstream labels.
- Add a two-layer MLP projection head after the encoder. The contrastive loss operates on projected features, while downstream models use encoder features.
- Use normalized embeddings and NT-Xent loss with a configurable temperature.
- Start with a high-memory-GPU reference configuration: global batch size 256 or larger when memory permits, mixed precision, LARS or AdamW optimizer, cosine learning-rate schedule, and a warmup phase.
- Train for a sufficiently long fixed schedule (reference: 200 epochs) after validating the pipeline with a short smoke run. Save best/last checkpoints and the exact configuration.

### X-ray-safe augmentation policy

SimCLR needs two independently augmented views of the same image, but X-ray transformations must preserve clinically meaningful anatomy.

- Resize to a larger intermediate size, then use random resized crops with a conservative crop scale.
- Use modest rotation and translation only.
- Use mild brightness and contrast variation to model acquisition differences.
- Use modest Gaussian blur or noise.
- Do not apply hue or saturation changes to grayscale radiographs.
- Avoid aggressive crops, large rotations, posterization, or cutout transforms that can remove pathology-bearing regions.
- Treat horizontal flipping as a configurable ablation. The default primary protocol should disable it because laterality can be clinically informative.

The project should save example pairs of augmented views before launching full training and manually inspect them for anatomical plausibility.

### Downstream models

Train three methods with identical data splits, input resolution, label mappings, and evaluation code:

1. **Supervised from scratch**: randomly initialized selected backbone plus classification head, trained only on the chosen labeled subset.
2. **SimCLR linear probe**: freeze the selected SimCLR-pretrained backbone and train only the five-logit classification head.
3. **SimCLR fine-tuning**: initialize the selected backbone from its SimCLR checkpoint, then optimize the encoder and classification head together on the labeled subset.

Use `BCEWithLogitsLoss` with per-label positive weighting computed from the labeled training subset. Store the weights in the run metadata. Optimize AUROC-oriented model selection with the development set; do not select checkpoints from the official validation set.

## 6. Experiment Matrix

### Vision-language model extension

Add a VLM track to determine whether image-text pretraining provides useful medical representations beyond image-only SimCLR. VLM results must be reported separately from SimCLR results because the VLM may have learned from external paired image-report data and therefore answers a different transfer-learning question.

Use CheXzero as the primary VLM baseline because it is a CLIP-style model designed for chest X-ray image-text alignment and naturally supports pathology prompts. Pin the exact checkpoint and implementation revision. If CheXzero cannot be used because of an unavailable checkpoint or incompatible license, use one radiology-domain image-text model with equivalent image-embedding and text-embedding interfaces, and identify it as a protocol deviation in the report. Record its name, revision, source, pretraining data description, license, image preprocessing, and any known CheXpert overlap risk before use. Do not silently substitute a general-purpose natural-image CLIP model for the primary VLM result.

The VLM track has three roles:

1. **Zero-shot prompt classification**: score each X-ray against pathology-present and pathology-absent text prompts without fitting a CheXpert classifier. This measures direct language-grounded transfer.
2. **Frozen VLM feature probe**: freeze the VLM image encoder and train the same five-logit linear head used for SimCLR. This isolates the quality of its visual representations under the project label budgets.
3. **VLM fine-tuning**: fine-tune the image encoder and classification head at the same label budgets when the checkpoint license and hardware permit. Use a lower encoder learning rate than the newly initialized head.

The primary image-only question remains SimCLR versus supervised learning from scratch. The VLM extension answers a secondary question: whether prior image-text alignment changes zero-shot performance, representation quality, or label efficiency relative to image-only SSL.

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

- Compare SimCLR, supervised, and VLM feature spaces using UMAP or t-SNE only as qualitative visualizations; do not treat visual separation as a performance metric.
- Produce class-conditioned retrieval panels: for selected validation queries, retrieve nearest training embeddings and inspect whether anatomy, acquisition artifacts, or pathology cues drive similarity. De-identify and keep examples local to the project.
- Analyze error slices by view position (`AP` versus `PA`), patient age group if available, sex if available, and label prevalence. Omit a slice when its sample size is too small for stable estimates.
- Use Grad-CAM or an equivalent image-attribution method for the classifier head. Mark every heatmap as a post-hoc explanation, not proof of clinical reasoning or localization.
- If a text-generation-capable VLM is used for narrative analysis, restrict it to templated, non-diagnostic descriptions of model outputs and require human review. It must not create ground-truth labels, alter metrics, or be presented as a clinical report generator.
- Maintain an error-analysis worksheet containing run ID, image ID, true labels, predicted probabilities, selected threshold, view, attribution artifact path, and a short reviewer observation.

### Label budgets and seeds

Run each downstream method at 1%, 5%, 10%, 25%, and 100% of labeled training patients. Use three fixed, published random seeds for each budget. All methods under a seed/budget pair must receive the exact same sampled patients.

The image-only core produces 45 downstream runs: 3 methods x 5 budgets x 3 seeds. The VLM extension adds 15 frozen-probe runs (5 budgets x 3 seeds) and, if compute permits, 9 fine-tuning runs (1%, 10%, and 100% x 3 seeds). Run the VLM zero-shot evaluation once per fixed prompt set because it has no sampled-label training phase. SimCLR pretraining may be run once per training partition and reused across downstream budgets; run an additional pretraining seed only after the primary matrix is complete.

### Recommended execution order

1. Verify dataset paths and generate prepared manifests.
2. Produce patient-disjoint partitions and persist their IDs.
3. Run unit tests and a 100- to 1,000-image smoke test for each data loader and training loop.
4. Train a supervised 100% baseline to establish a functioning end-to-end reference.
5. Inspect SimCLR augmentation pairs, then run a short pretraining validation run.
6. Complete full SimCLR pretraining and archive its checkpoint.
7. Run linear-probe and fine-tuning experiments from smallest to largest label budget.
8. Validate VLM preprocessing and prompt scoring on the development split; then run the zero-shot and frozen-feature experiments.
9. Run VLM fine-tuning only after the frozen-feature and zero-shot results are complete and only at the designated budget tiers.
10. Re-run failed or anomalous jobs only with a recorded reason.
11. Lock selected configurations, evaluate once on official validation data, and generate the final report.

### Initial hyperparameter defaults

These are starting points, not results to tune against the official validation set.

| Component | Initial configuration |
| --- | --- |
| Image size | 224 x 224 |
| SSL backbones | ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, ViT-B/16 |
| SSL epochs | 200 |
| SSL temperature | 0.1 to 0.2, selected on development protocol |
| SSL effective batch size | 256+ with gradient accumulation if required |
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
- Patient-level bootstrap 95% confidence intervals for final AUROC and for the fine-tuned-SimCLR minus supervised-baseline difference.
- Training time, peak GPU memory where available, epochs completed, and checkpoint size.
- Zero-shot VLM metrics with the frozen prompt-set version and calibration status clearly identified.
- Pairwise label-efficiency deltas at each budget: SimCLR fine-tuning minus supervised training, and VLM frozen probe minus SimCLR linear probe.
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
4. The two-view SimCLR dataset returns distinct but anatomically plausible tensor views with the expected shape and finite values.
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

### Milestone 2: Baseline

The supervised 100% run completes, produces development metrics, and can be evaluated reproducibly from its checkpoint.

### Milestone 3: SSL representation

SimCLR pretraining completes with inspected augmentations, finite contrastive loss, saved checkpoints, and logged configuration.

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
| Large SimCLR batch does not fit memory | Use gradient accumulation, mixed precision, and a configured lower batch size while preserving the global-batch target where possible. |
| Repeated validation-set tuning overfits results | Use the internal development split for all selection and evaluate the official validation manifest only after configurations are locked. |
| Small low-budget subsets are unstable | Use fixed multi-seed patient samples and report variance, not just the best run. |
| VLM pretrained on data overlapping CheXpert | Review the model card and pretraining documentation; disclose known or unresolved overlap and label results as transfer benchmarks, not independent generalization. |
| Prompt wording changes zero-shot score | Pre-register a small versioned template set and tune only on the internal development split. |
| VLM is too expensive to fine-tune | Complete zero-shot and frozen-feature probes first; use parameter-efficient fine-tuning only if it is clearly reported as such. |
| Interpretability artifacts are over-read | Present retrievals and attributions as qualitative evidence of model behavior, never as clinical validation. |

## 12. Final Deliverables

1. A documented, configuration-driven training pipeline.
2. Patient-level manifests and reproducible split files.
3. Supervised, linear-probe, and fine-tuned SimCLR checkpoints and metric logs.
4. Versioned VLM prompt sets, zero-shot scores, frozen-feature probes, and optional VLM fine-tuning artifacts.
5. Aggregated tables and plots for the image-only and image-text label-efficiency comparisons.
6. A final technical report that states the question, methods, results, VLM provenance, limitations, and conditions needed to reproduce the study.

## 13. Implementation Status (2026-10-02)

The repository contains the configured data and training pipeline. The image-only
training pipeline has not yet been run end to end; no performance claim,
checkpoint, split artifact, or final-validation result is implied by code
presence alone.

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
- Each training/evaluation output receives `run_metadata.json` with the full
  configuration, command, Git revision, Python/PyTorch versions, CUDA status,
  hardware identifier, and timestamp.

### Implemented image-only experiments

- `scripts/train_pretrain.py` trains the selected backbone with SimCLR, writes conservative
  two-view augmentation samples, and saves `best.pt`, `last.pt`, and loss
  history. Horizontal flip is explicitly configurable and disabled by default.
- `scripts/prepare_smoke_manifest.py` creates a bounded unlabeled manifest for
  a short SimCLR pipeline check. `configs/pretrain/simclr_smoke.yaml` uses it
  with ResNet-18 and one epoch before full pretraining is started.
- `scripts/train_downstream.py` supports `supervised`, `simclr_linear`, and
  `simclr_finetune` modes, accepts persisted sampled-patient JSON files, uses
  subset-specific positive weights, selects checkpoints by development macro
  AUROC, applies early stopping, and persists development-selected thresholds.
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

### Architecture comparison

- `configs/models/backbones.yaml` declares the shared comparison set:
  ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, and ViT-B/16.
- The encoder factory removes each classifier head and returns a fixed feature
  vector interface used by SimCLR and the five-logit downstream classifier.
  ViT-B/16 contributes its class-token feature vector.
- The three image-only downstream methods run with the same architecture set,
  persisted patient cohorts, label budgets, and seeds. Aggregated metrics retain
  the architecture as a separate comparison field.
- Image-only training resumes from the latest epoch checkpoint when available,
  allowing long runs to continue after an interruption.

### Deferred implementation and operational requirements

- Gradient accumulation, warmup, LARS, iterative multilabel stratification,
  run-duration/peak-memory logging, final-AUROC paired-difference bootstrap,
  calibration, plot generation, retrieval, UMAP/t-SNE, and Grad-CAM are not
  implemented yet.
- The existing unit tests cover uncertainty conversion, patient-overlap checks,
  patient budgets, NT-Xent finiteness, and basic multi-label metrics. The
  remaining checks in Section 9 still need implementation.
- Before any expensive run, inspect `augmentation_pairs.png`, execute the test
  suite and smoke runs, review VLM provenance/license/overlap risk, and lock a
  configuration on the development split before calling final evaluation.
