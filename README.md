# CheXpert SSL — run the notebook

Open [chexpert_ssl.ipynb](chexpert_ssl.ipynb) in local Jupyter Notebook,
JupyterLab or VS Code, review its YAML settings, and choose **Run All**.
The notebook contains the implementation and executes the study in order.
There are no cell tags or `RUN_*` switches.

## Setup

Use a Python 3.10–3.13 kernel. The first cell locates the notebook and makes its
directory the project root. Missing project dependencies are installed into the
current kernel automatically. For CUDA, install the appropriate PyTorch build
in that environment first. Put the dataset in `CheXpert-v1.0-small/` beside the
notebook, or change `dataset_root` in the embedded YAML. If the kernel starts
outside the project, set `CHEXPERT_PROJECT_ROOT` to the notebook directory.

Optional terminal setup:

```bash
python -m pip install '.[dev,notebook]'
jupyter lab chexpert_ssl.ipynb
```

## What Run All does

1. Loads setup, YAML settings and implementation.
2. Plans the experiment matrix and prepares or validates/reuses saved cohorts.
3. Runs tests and bounded smoke checks.
4. Displays augmentation pairs and asks you to type `reviewed` after inspection.
5. Pretrains all five SSL methods and trains the downstream matrix.
6. Aggregates development results and asks for a development-only selection reason.
7. Locks the declared study arms, evaluates held-out data, and aggregates final results.

The default matrix is **5 pretraining runs and 180 downstream runs**: SimCLR,
MoCo v2, BYOL, NNCLR and SwAV with probes/fine-tuning, plus scratch and pinned
ImageNet baselines across five budgets and three seeds. Edit the YAML before
running to change the study. ImageNet smoke needs access to its pinned weights.

The two review prompts preserve the scientific protocol in
[docs/context.md](docs/context.md). A matching saved augmentation review and
verified selection locks are reused on reruns. Empty/refused review input stops
execution before the dependent stage. Preparation reuses cohorts without
resampling; mismatched or incomplete artifacts fail rather than being overwritten.
Compatible completed runs and final results are reused by the runner.

## Configuration and artifacts

`CONFIG_YAML` holds the settings. `configs/...` names are embedded configuration
keys, not filesystem dependencies. The primary profile is
`configs/experiments/notebook_reference.yaml`; the historical seed-42 profile
remains available separately. External YAML/JSON overrides are supported.

```text
chexpert_ssl.ipynb           executable notebook and configuration
pyproject.toml              dependency and verification settings
docs/context.md             scientific protocol
tests/                      regression tests and test-only notebook loader
CheXpert-v1.0-small/         local raw dataset (ignored)
data/processed/             manifests, shared patient cohorts, caches (ignored)
outputs/                   checkpoints, plans, metrics, reports (ignored)
```

Reference data is under `data/processed/notebook-reference/`. New results are
under `outputs/notebook-consolidated/`, with plans under
`outputs/plans/notebook-consolidated/`. Older artifacts are retained.
Notebook loaders use `num_workers: 0`; cache decoding still uses threads.

Save code changes and restart the kernel before rerunning. Executable notebook
code enters resume signatures; changed code/settings require new output and
plan directories. Existing package/original-notebook checkpoints are not
automatically compatible resume points.

## Individual operations and verification

For a targeted rerun, call `run_stage(...)` from a cell, for example:

```python
run_stage("downstream", methods=["simclr"], budgets=[0.01], seeds=[0])
run_stage("aggregate", partition="development")
```

`execute_task(["analyze_errors", "--help"])` describes error worksheets;
`execute_task(["repair_pretrain_manifest", "--help"])` describes legacy repair.
VLM functions remain experimental and outside the default Run All study. They
require `.[vlm]`, a reviewed radiology checkpoint/revision, license and overlap
review, and explicit `image-text VLM` reporting. No general-purpose CLIP fallback.

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Tests load setup/definitions without launching real-data experiments. They also
exercise the Run All sequence with substituted tasks and a synthetic
training/locked-evaluation pipeline in fresh Jupyter kernels. Synthetic CPU
checks do not establish GPU throughput or clinical performance. Outputs must
not be used as diagnosis or clinical advice.
