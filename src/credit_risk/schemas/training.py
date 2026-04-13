"""Schema hints for training dataset validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSchema:
    required_columns: tuple[str, ...] = (
        "application_id",
        "annual_income",
        "loan_amount",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "revolving_utilization",
        "open_accounts",
        "delinquencies_2y",
        "default_flag",
    )

    numeric_columns: tuple[str, ...] = (
        "annual_income",
        "loan_amount",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "revolving_utilization",
        "open_accounts",
        "delinquencies_2y",
    )
