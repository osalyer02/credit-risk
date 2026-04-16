"""FastAPI routes for health checks and credit risk scoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from credit_risk.data.load import load_dataset
from credit_risk.schemas.payloads import (
    ApplicantInput,
    BatchPredictionResponse,
    HealthResponse,
    PredictBatchRequest,
    PredictionResponse,
)
from credit_risk.scoring.predict import PredictionService

router = APIRouter()


def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not initialized. Train or load artifacts first.",
        )
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "prediction_service", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    config = getattr(request.app.state, "config", None)

    if service is not None:
        response = service.health()
        response.model_loaded = True
        response.startup_error = None
        return response

    model_version = "unavailable"
    if config is not None:
        model_version = config.project.model_version

    return HealthResponse(
        model_version=model_version,
        model_loaded=False,
        startup_error=startup_error,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    applicant: ApplicantInput,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return service.predict_one(applicant)


@router.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: PredictBatchRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    applicants: list[ApplicantInput] = []

    if payload.applicants:
        applicants = payload.applicants
    elif payload.s3_input_path:
        frame = load_dataset(payload.s3_input_path)
        expected_fields = set(ApplicantInput.model_fields)
        rows: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            rows.append({key: value for key, value in row.items() if key in expected_fields})
        applicants = [ApplicantInput.model_validate(row) for row in rows]

    if not applicants:
        raise HTTPException(status_code=400, detail="No applicants available for batch scoring")

    return service.predict_batch(applicants)


@router.get("/prediction/{request_id}")
def get_prediction(
    request_id: str,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, Any]:
    record = service.fetch_prediction(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record
