from __future__ import annotations

import pandas as pd

from credit_risk.features.engineering import apply_feature_engineering


def test_feature_engineering_creates_derived_columns():
    frame = pd.DataFrame(
        {
            "application_id": ["a-1"],
            "annual_income": [100000],
            "loan_amount": [25000],
            "fico_range_low": [680],
            "fico_range_high": [700],
            "revolving_utilization": [58.0],
            "delinquencies_2y": [1],
        }
    )

    engineered = apply_feature_engineering(frame)

    assert "loan_to_income" in engineered.columns
    assert "credit_score_mid" in engineered.columns
    assert "utilization_bucket" in engineered.columns
    assert "has_recent_delinquency" in engineered.columns
    assert engineered.loc[0, "loan_to_income"] == 0.25
    assert engineered.loc[0, "credit_score_mid"] == 690
    assert engineered.loc[0, "has_recent_delinquency"] == 1
