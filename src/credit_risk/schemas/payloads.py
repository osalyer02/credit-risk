"""Pydantic request/response schemas used by API and scoring services."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ApplicantInput(BaseModel):
    application_id: str = Field(min_length=1)
    annual_income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    dti: float = Field(ge=0)
    fico_range_low: int = Field(ge=300, le=850)
    fico_range_high: int = Field(ge=300, le=850)
    revolving_utilization: float = Field(ge=0)
    open_accounts: int = Field(ge=0)
    delinquencies_2y: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_fico_bounds(self) -> "ApplicantInput":
        if self.fico_range_high < self.fico_range_low:
            raise ValueError("fico_range_high must be greater than or equal to fico_range_low")
        return self


class PredictBatchRequest(BaseModel):
    applicants: Optional[list[ApplicantInput]] = None
    s3_input_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_source(self) -> "PredictBatchRequest":
        has_applicants = bool(self.applicants)
        has_s3 = bool(self.s3_input_path)
        if has_applicants == has_s3:
            raise ValueError("Exactly one of applicants or s3_input_path must be provided")
        return self


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    pd_score: float = Field(ge=0.0, le=1.0)
    risk_band: str
    decision_recommendation: Literal["APPROVE", "REVIEW", "DECLINE"]
    reason_codes: list[str]
    latency_ms: int = Field(ge=0)


class BatchPredictionResponse(BaseModel):
    batch_id: str
    request_count: int = Field(ge=0)
    output_s3_path: Optional[str] = None
    prediction_ids: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "credit-risk-platform"
    model_version: str
    model_loaded: bool = True
    startup_error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictionRecord(BaseModel):
    request_id: str
    application_id: str
    model_version: str
    pd_score: float
    risk_band: str
    decision: str
    reason_codes: list[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("pd_score")
    @classmethod
    def validate_pd(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("pd_score must be between 0 and 1")
        return value
