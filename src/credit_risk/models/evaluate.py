"""Evaluation utilities for binary credit default models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def confusion_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []

    for threshold in thresholds:
        predicted = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        summaries.append(
            {
                "threshold": float(threshold),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
            }
        )

    return summaries


def calibration_outputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    ece = 0.0
    total = len(y_prob)

    for idx in range(n_bins):
        mask = bin_ids == idx
        if np.any(mask):
            observed = float(np.mean(y_true[mask]))
            predicted = float(np.mean(y_prob[mask]))
            weight = float(np.sum(mask) / total)
            ece += abs(observed - predicted) * weight

    return {
        "prob_true": [float(value) for value in prob_true],
        "prob_pred": [float(value) for value in prob_pred],
        "ece": float(ece),
    }


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float] = (0.5,),
) -> dict[str, object]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ks_statistic": ks_statistic(y_true, y_prob),
        "confusion_matrices": confusion_summary(y_true, y_prob, thresholds),
        "calibration": calibration_outputs(y_true, y_prob),
    }
