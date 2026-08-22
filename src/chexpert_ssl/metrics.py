"""Multi-label metrics with explicit handling for single-class labels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
