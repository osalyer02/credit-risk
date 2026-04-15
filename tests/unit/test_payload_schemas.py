from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from credit_risk.schemas.payloads import ApplicantInput, PredictionResponse
from credit_risk.schemas.registry import ModelRegistryRecord


def test_applicant_schema_validates_required_fields():
    applicant = ApplicantInput(
        application_id="app-1",
        annual_income=85000,
        loan_amount=12000,
        dti=18.7,
        fico_range_low=680,
        fico_range_high=684,
        revolving_utilization=42.1,
        open_accounts=8,
        delinquencies_2y=0,
    )

    assert applicant.application_id == "app-1"


def test_applicant_schema_rejects_inverted_fico_range():
    with pytest.raises(ValidationError):
        ApplicantInput(
            application_id="app-1",
            annual_income=85000,
            loan_amount=12000,
            dti=18.7,
            fico_range_low=700,
            fico_range_high=680,
            revolving_utilization=42.1,
            open_accounts=8,
            delinquencies_2y=0,
        )


def test_prediction_response_schema_bounds_pd_score():
    response = PredictionResponse(
        request_id="req-1",
        model_version="v1",
        pd_score=0.42,
        risk_band="C",
        decision_recommendation="REVIEW",
        reason_codes=["dti increased default risk"],
        latency_ms=23,
    )
    assert response.pd_score == 0.42


def test_model_registry_record_schema():
    record = ModelRegistryRecord(
        model_id="credit-risk-platform:v1:logistic_regression",
        model_version="v1",
        model_type="logistic_regression",
        calibration_method="isotonic",
        artifact_uri="artifacts/models/v1/model_bundle.joblib",
        metrics_uri="artifacts/metrics/v1/metrics.json",
        training_data_path="data/raw/credit_train.csv",
        feature_columns=["annual_income", "loan_amount", "dti"],
        status="TRAINED",
        created_at=datetime.now(timezone.utc),
    )
    assert record.model_type == "logistic_regression"
