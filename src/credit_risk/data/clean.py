"""Data cleaning helpers for training and scoring workflows."""

from __future__ import annotations

import pandas as pd


def clean_training_data(
    frame: pd.DataFrame,
    target_column: str = "default_flag",
    id_column: str = "application_id",
) -> pd.DataFrame:
    """Apply deterministic cleaning and minimal data quality safeguards."""

    cleaned = frame.copy()

    if id_column in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[id_column], keep="last")

    cleaned = cleaned.dropna(subset=[target_column])

    non_negative_columns = [
        "annual_income",
        "loan_amount",
        "dti",
        "revolving_utilization",
        "open_accounts",
        "delinquencies_2y",
    ]

    for column in non_negative_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].clip(lower=0)

    if "revolving_utilization" in cleaned.columns:
        cleaned["revolving_utilization"] = cleaned["revolving_utilization"].clip(upper=200)

    return cleaned.reset_index(drop=True)
