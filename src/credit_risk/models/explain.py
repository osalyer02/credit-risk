"""Explainability utilities with SHAP-first and deterministic fallbacks."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def _unwrap_pipeline(estimator: Any) -> Any:
    if hasattr(estimator, "calibrated_classifiers_"):
        calibrated = estimator.calibrated_classifiers_[0]
        return calibrated.estimator
    return estimator


def _get_preprocessor_and_classifier(estimator: Any) -> tuple[Any, Any]:
    base = _unwrap_pipeline(estimator)
    if hasattr(base, "named_steps"):
        return base.named_steps["preprocessor"], base.named_steps["classifier"]
    raise TypeError("Estimator does not expose preprocessor/classifier steps")


def _friendly_feature_name(raw_name: str) -> str:
    name = raw_name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ")


def _feature_names(preprocessor: Any) -> list[str]:
    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]


def global_feature_importance(
    estimator: Any,
    sample_frame: pd.DataFrame,
    top_k: int = 20,
) -> list[dict[str, Union[float, str]]]:
    """Generate global feature importance records."""

    preprocessor, classifier = _get_preprocessor_and_classifier(estimator)
    transformed = preprocessor.transform(sample_frame)
    feature_names = _feature_names(preprocessor)

    importances: np.ndarray

    try:
        import shap

        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            importances = np.mean(np.abs(shap_values), axis=0)
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = np.ones(transformed.shape[1])
    except Exception:
        if hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        elif hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_)
        else:
            importances = np.ones(transformed.shape[1])

    ranking = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "feature": _friendly_feature_name(feature_names[idx]),
            "importance": float(importances[idx]),
        }
        for idx in ranking
    ]


def reason_codes_for_row(
    estimator: Any,
    applicant_frame: pd.DataFrame,
    top_k: int = 3,
) -> list[str]:
    """Create local reason codes for a single applicant prediction."""

    preprocessor, classifier = _get_preprocessor_and_classifier(estimator)
    transformed = preprocessor.transform(applicant_frame)
    feature_names = _feature_names(preprocessor)

    contributions: Optional[np.ndarray] = None

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        row_vector = transformed[0]
        contributions = coefficients * row_vector
    else:
        try:
            import shap

            explainer = shap.Explainer(classifier)
            shap_values = explainer(transformed)
            contributions = np.asarray(shap_values.values[0])
        except Exception:
            contributions = None

    if contributions is None:
        if hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_)
            top_idx = np.argsort(importances)[::-1][:top_k]
            return [
                f"key factor: {_friendly_feature_name(feature_names[idx])}" for idx in top_idx
            ]
        return ["insufficient explanation data"]

    top_idx = np.argsort(np.abs(contributions))[::-1][:top_k]
    codes: list[str] = []
    for idx in top_idx:
        direction = "increased" if contributions[idx] >= 0 else "decreased"
        feature_name = _friendly_feature_name(feature_names[idx])
        codes.append(f"{feature_name} {direction} default risk")

    return codes
