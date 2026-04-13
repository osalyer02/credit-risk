"""Feature engineering transformations shared by train and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _utilization_bucket(utilization: pd.Series) -> pd.Series:
    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels = ["very_low", "low", "medium", "high", "very_high"]
    return pd.cut(utilization, bins=bins, labels=labels).astype("string")


def apply_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Create domain-inspired credit risk features."""

    enriched = frame.copy()

    annual_income = enriched.get("annual_income", pd.Series(index=enriched.index, dtype=float)).replace(
        {0: np.nan}
    )
    loan_amount = enriched.get("loan_amount", pd.Series(index=enriched.index, dtype=float))

    enriched["loan_to_income"] = (loan_amount / annual_income).fillna(0.0).clip(lower=0.0)
    enriched["credit_score_mid"] = (
        enriched.get("fico_range_low", 0) + enriched.get("fico_range_high", 0)
    ) / 2.0

    if "revolving_utilization" in enriched.columns:
        enriched["utilization_bucket"] = _utilization_bucket(enriched["revolving_utilization"])
    else:
        enriched["utilization_bucket"] = "unknown"

    if "delinquencies_2y" in enriched.columns:
        enriched["has_recent_delinquency"] = (enriched["delinquencies_2y"] > 0).astype(int)
    else:
        enriched["has_recent_delinquency"] = 0

    return enriched
