"""Dataframe-level schema validation for training data."""

from __future__ import annotations

import pandas as pd

from credit_risk.schemas.training import TrainingSchema


class DataValidationError(RuntimeError):
    """Raised when dataset schema validation fails."""


def validate_training_schema(
    frame: pd.DataFrame,
    target_column: str = "default_flag",
    id_column: str = "application_id",
) -> pd.DataFrame:
    """Validate expected columns and coerce numeric training features."""

    schema = TrainingSchema()
    alias_map = {
        "default_flag": target_column,
        "application_id": id_column,
    }

    required_columns = tuple(alias_map.get(column, column) for column in schema.required_columns)

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    validated = frame.copy()

    numeric_columns = [alias_map.get(column, column) for column in schema.numeric_columns]

    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    validated[target_column] = pd.to_numeric(validated[target_column], errors="coerce")

    if validated[target_column].isna().any():
        raise DataValidationError(f"Target column {target_column} contains invalid values")

    invalid_targets = set(validated[target_column].dropna().unique()) - {0, 1}
    if invalid_targets:
        raise DataValidationError(f"Target column must be binary (0/1), found: {sorted(invalid_targets)}")

    return validated
