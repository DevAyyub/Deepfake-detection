"""sfdet.metrics.classification — detection metrics.

AUC-ROC is the ANCHOR metric (threshold-free; every results-table row is
compared in AUC). Accuracy (at a 0.5 threshold by default) and EER (from the ROC)
are reported alongside. These are the primitives; the step-7 evaluator computes
them per dataset at both frame and video granularity. The training loop uses
them for validation / checkpoint selection.

Single-class inputs (a batch/video with only one label present) return NaN for
AUC/EER rather than raising, so logging never crashes on a degenerate split.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def accuracy(labels, scores, threshold: float = 0.5) -> float:
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return float((preds == labels).mean())


def eer(labels, scores) -> float:
    """Equal error rate: the operating point where FPR == FNR (read off the ROC)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict:
    """All three at once: {'auc', 'acc', 'eer'}."""
    return {
        "auc": roc_auc(labels, scores),
        "acc": accuracy(labels, scores, threshold),
        "eer": eer(labels, scores),
    }
