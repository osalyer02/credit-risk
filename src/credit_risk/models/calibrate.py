"""Probability calibration routines for trained classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from credit_risk.models.evaluate import compute_binary_classification_metrics

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - compatibility branch
    FrozenEstimator = None  # type: ignore[assignment]


@dataclass
class CalibrationResult:
    method: str
    estimator: object
    metrics: dict[str, object]


def _calibration_score(metrics: dict[str, object]) -> float:
    calibration = metrics.get("calibration", {})
    ece = float(calibration.get("ece", 1.0))
    brier = float(metrics["brier_score"])
    ll = float(metrics["log_loss"])
    return brier + ll + ece


def calibrate_estimator(
    estimator: object,
    x_val,
    y_val: np.ndarray,
    methods: list[str],
) -> tuple[CalibrationResult, list[CalibrationResult]]:
    """Fit candidate calibration wrappers and return the best."""

    candidates: list[CalibrationResult] = []

    for method in methods:
        if FrozenEstimator is not None:
            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(estimator),
                method=method,
            )
        else:
            try:
                calibrator = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
            except TypeError:
                calibrator = CalibratedClassifierCV(
                    base_estimator=estimator,
                    method=method,
                    cv="prefit",
                )
        calibrator.fit(x_val, y_val)

        probabilities = calibrator.predict_proba(x_val)[:, 1]
        metrics = compute_binary_classification_metrics(y_val, probabilities)

        candidates.append(
            CalibrationResult(
                method=method,
                estimator=calibrator,
                metrics=metrics,
            )
        )

    if not candidates:
        raise RuntimeError("No calibration candidates were generated")

    best = min(candidates, key=lambda candidate: _calibration_score(candidate.metrics))
    return best, candidates
