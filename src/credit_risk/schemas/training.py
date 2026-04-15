"""Schema hints for training dataset validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Union

from pydantic import BaseModel, Field


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


class SchemaValidationReport(BaseModel):
    """Structured validation report for incoming training datasets."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    required_columns: list[str]
    missing_columns: list[str] = Field(default_factory=list)
    coerced_to_null_counts: dict[str, int] = Field(default_factory=dict)
    target_invalid_count: int = Field(ge=0, default=0)
    target_unique_values: list[Union[int, float]] = Field(default_factory=list)
    passed: bool = False
