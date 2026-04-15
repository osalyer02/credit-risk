"""Dataframe-level schema validation for training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from credit_risk.schemas.training import SchemaValidationReport, TrainingSchema


class DataValidationError(RuntimeError):
    """Raised when dataset schema validation fails."""

    def __init__(self, message: str, report: Optional[SchemaValidationReport] = None) -> None:
        super().__init__(message)
        self.report = report


def _write_report(report: SchemaValidationReport, report_path: Optional[Union[str, Path]]) -> None:
    if report_path is None:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")


def validate_training_schema_with_report(
    frame: pd.DataFrame,
    target_column: str = "default_flag",
    id_column: str = "application_id",
    report_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, SchemaValidationReport]:
    """Validate expected columns, coerce types, and emit a validation report."""

    schema = TrainingSchema()
    alias_map = {
        "default_flag": target_column,
        "application_id": id_column,
    }

    required_columns = [alias_map.get(column, column) for column in schema.required_columns]
    missing_columns = [column for column in required_columns if column not in frame.columns]

    report = SchemaValidationReport(
        total_rows=int(len(frame)),
        total_columns=int(len(frame.columns)),
        required_columns=required_columns,
        missing_columns=missing_columns,
    )

    if missing_columns:
        report.passed = False
        _write_report(report, report_path)
        raise DataValidationError(f"Missing required columns: {missing_columns}", report=report)

    validated = frame.copy()
    numeric_columns = [alias_map.get(column, column) for column in schema.numeric_columns]

    coerced_to_null_counts: dict[str, int] = {}
    for column in numeric_columns:
        before_non_null = validated[column].notna()
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        newly_null = int((before_non_null & validated[column].isna()).sum())
        coerced_to_null_counts[column] = newly_null

    target_before_non_null = validated[target_column].notna()
    validated[target_column] = pd.to_numeric(validated[target_column], errors="coerce")
    target_newly_null = int((target_before_non_null & validated[target_column].isna()).sum())

    invalid_targets = sorted(set(validated[target_column].dropna().unique()) - {0, 1})
    target_invalid_count = int(target_newly_null + len(invalid_targets))

    report.coerced_to_null_counts = coerced_to_null_counts
    report.target_invalid_count = target_invalid_count
    report.target_unique_values = sorted([float(value) for value in validated[target_column].dropna().unique()])

    if target_newly_null > 0:
        report.passed = False
        _write_report(report, report_path)
        raise DataValidationError(
            f"Target column {target_column} contains invalid values",
            report=report,
        )

    if invalid_targets:
        report.passed = False
        _write_report(report, report_path)
        raise DataValidationError(
            f"Target column must be binary (0/1), found: {invalid_targets}",
            report=report,
        )

    report.passed = True
    _write_report(report, report_path)

    return validated, report


def validate_training_schema(
    frame: pd.DataFrame,
    target_column: str = "default_flag",
    id_column: str = "application_id",
) -> pd.DataFrame:
    """Validate expected columns and return a validated dataframe."""

    validated, _ = validate_training_schema_with_report(
        frame=frame,
        target_column=target_column,
        id_column=id_column,
    )
    return validated
