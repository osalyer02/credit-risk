from __future__ import annotations

import json

import pandas as pd
import pytest

from credit_risk.data.validate import (
    DataValidationError,
    validate_training_schema,
    validate_training_schema_with_report,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_id": ["app-1", "app-2"],
            "annual_income": ["85000", "91000"],
            "loan_amount": ["12000", "17000"],
            "dti": ["18.7", "16.4"],
            "fico_range_low": [680, 700],
            "fico_range_high": [684, 705],
            "revolving_utilization": ["42.1", "35.0"],
            "open_accounts": ["8", "10"],
            "delinquencies_2y": [0, 1],
            "default_flag": [0, 1],
        }
    )


def test_validate_training_schema_with_report_writes_report(tmp_path):
    frame = _sample_frame()
    report_path = tmp_path / "validation_report.json"

    validated, report = validate_training_schema_with_report(frame, report_path=report_path)

    assert validated["annual_income"].dtype.kind in {"f", "i"}
    assert report.passed is True
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["missing_columns"] == []


def test_validate_training_schema_raises_on_missing_required_column(tmp_path):
    frame = _sample_frame().drop(columns=["dti"])
    report_path = tmp_path / "validation_report.json"

    with pytest.raises(DataValidationError):
        validate_training_schema_with_report(frame, report_path=report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "dti" in payload["missing_columns"]


def test_validate_training_schema_wrapper_returns_frame():
    validated = validate_training_schema(_sample_frame())
    assert len(validated) == 2
