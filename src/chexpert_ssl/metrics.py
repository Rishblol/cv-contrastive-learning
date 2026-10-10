"""Multi-label metrics with explicit handling for single-class labels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .data import TARGETS


def multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape")
    if targets.shape[1] != len(TARGETS):
        raise ValueError(f"Expected {len(TARGETS)} label columns")

    results: dict[str, float] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    for index, label in enumerate(TARGETS):
        truth, scores = targets[:, index], probabilities[:, index]
        if len(np.unique(truth)) < 2:
            results[f"auroc_{label}"] = float("nan")
            results[f"auprc_{label}"] = float("nan")
            continue
        auroc = float(roc_auc_score(truth, scores))
        auprc = float(average_precision_score(truth, scores))
        results[f"auroc_{label}"] = auroc
        results[f"auprc_{label}"] = auprc
        aurocs.append(auroc)
        auprcs.append(auprc)
    results["macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    results["macro_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")
    return results


def select_thresholds(
    targets: np.ndarray, probabilities: np.ndarray, method: str = "balanced_accuracy"
) -> dict[str, float]:
    """Choose per-label balanced-accuracy thresholds on development data only."""
    thresholds: dict[str, float] = {}
    for index, label in enumerate(TARGETS):
        truth, scores = targets[:, index], probabilities[:, index]
        if len(np.unique(truth)) < 2:
            thresholds[label] = 0.5
            continue
        if method == "youden":
            fpr, tpr, values = roc_curve(truth, scores)
            value = values[int(np.argmax(tpr - fpr))]
            thresholds[label] = float(value) if np.isfinite(value) else 0.5
            continue
        if method != "balanced_accuracy":
            raise ValueError("Unknown threshold selection method")
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
        values = [balanced_accuracy_score(truth, scores >= value) for value in candidates]
        thresholds[label] = float(candidates[int(np.argmax(values))])
    return thresholds


def threshold_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    notebook: bool = False,
) -> dict[str, float]:
    """Return thresholded metrics with safe NaNs for single-class labels."""
    result: dict[str, float] = {}
    f1s: list[float] = []
    balanced: list[float] = []
    for index, label in enumerate(TARGETS):
        truth = targets[:, index]
        predicted = probabilities[:, index] >= thresholds[label]
        if notebook:
            positive = truth == 1
            negative = ~positive
            tp, fn = int((predicted & positive).sum()), int((~predicted & positive).sum())
            tn, fp = int((~predicted & negative).sum()), int((predicted & negative).sum())
            sensitivity = tp / (tp + fn) if tp + fn else float("nan")
            specificity = tn / (tn + fp) if tn + fp else float("nan")
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
            rates = [v for v in (sensitivity, specificity) if np.isfinite(v)]
            balanced_score = float(np.mean(rates)) if rates else float("nan")
            result.update(
                {
                    f"sensitivity_{label}": sensitivity,
                    f"specificity_{label}": specificity,
                    f"f1_{label}": f1,
                    f"balanced_accuracy_{label}": balanced_score,
                }
            )
            if np.isfinite(f1):
                f1s.append(f1)
            if np.isfinite(balanced_score):
                balanced.append(balanced_score)
            continue
        if len(np.unique(truth)) < 2:
            for metric in ("sensitivity", "specificity", "f1", "balanced_accuracy"):
                result[f"{metric}_{label}"] = float("nan")
            continue
        positives = truth == 1
        negatives = ~positives
        result[f"sensitivity_{label}"] = float(predicted[positives].mean())
        result[f"specificity_{label}"] = float((~predicted[negatives]).mean())
        result[f"f1_{label}"] = float(f1_score(truth, predicted, zero_division=0))
        result[f"balanced_accuracy_{label}"] = float(balanced_accuracy_score(truth, predicted))
        f1s.append(result[f"f1_{label}"])
        balanced.append(result[f"balanced_accuracy_{label}"])
    result["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    result["macro_balanced_accuracy"] = float(np.mean(balanced)) if balanced else float("nan")
    return result


def patient_bootstrap_macro_auroc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    seed: int,
    samples: int = 1000,
) -> tuple[float, float]:
    """Patient-resampled percentile CI for macro AUROC."""
    patient_ids = np.asarray(patient_ids).astype(str)
    unique = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(patient_ids == patient) for patient in selected])
        score = multilabel_metrics(targets[indices], probabilities[indices])["macro_auroc"]
        if np.isfinite(score):
            values.append(score)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def patient_bootstrap_auroc(targets, probabilities, patient_ids, seed, samples=1000):
    """Notebook-style macro and pathology CIs from the same patient resamples."""
    patient_ids = np.asarray(patient_ids).astype(str)
    groups = [np.flatnonzero(patient_ids == p) for p in np.unique(patient_ids)]
    rng = np.random.RandomState(seed)
    values = {key: [] for key in ["macro_auroc", *(f"auroc_{l}" for l in TARGETS)]}
    for _ in range(samples):
        indices = np.concatenate([groups[k] for k in rng.randint(0, len(groups), len(groups))])
        metrics = multilabel_metrics(targets[indices], probabilities[indices])
        for key, collected in values.items():
            if np.isfinite(metrics[key]):
                collected.append(metrics[key])
    intervals = {
        key: tuple(float(v) for v in np.percentile(scores, [2.5, 97.5]))
        if scores
        else (float("nan"), float("nan"))
        for key, scores in values.items()
    }
    return {
        "macro_auroc": intervals["macro_auroc"],
        "per_label": {label: intervals[f"auroc_{label}"] for label in TARGETS},
    }
