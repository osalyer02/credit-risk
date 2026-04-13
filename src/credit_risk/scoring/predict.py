"""Prediction service for online and batch credit risk scoring."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import json

import pandas as pd

from credit_risk.aws.dynamodb import PredictionStore, create_prediction_store
from credit_risk.aws.s3 import ArtifactStore
from credit_risk.config.settings import load_config
from credit_risk.features.engineering import apply_feature_engineering
from credit_risk.models.explain import reason_codes_for_row
from credit_risk.schemas.config import AppConfig
from credit_risk.schemas.payloads import (
    ApplicantInput,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRecord,
    PredictionResponse,
)
from credit_risk.scoring.rules import assign_risk_band, decision_recommendation
from credit_risk.utils.artifacts import ModelBundle, create_artifact_store, load_bundle
from credit_risk.utils.logging import get_logger, log_event


logger = get_logger(__name__)


@dataclass
class PredictionService:
    """Loads the model bundle and serves inference operations."""

    config: AppConfig
    bundle: ModelBundle
    artifact_store: ArtifactStore
    prediction_store: PredictionStore

    @classmethod
    def from_config(cls, config: AppConfig) -> "PredictionService":
        artifact_store = create_artifact_store(config)
        bundle_key = f"{config.project.model_version}/{config.artifacts.model_filename}"
        bundle = load_bundle(artifact_store, bundle_key)
        prediction_store = create_prediction_store(config)
        return cls(
            config=config,
            bundle=bundle,
            artifact_store=artifact_store,
            prediction_store=prediction_store,
        )

    def health(self) -> HealthResponse:
        return HealthResponse(model_version=self.bundle.model_version)

    def _predict_probability(self, applicant_df: pd.DataFrame) -> float:
        probabilities = self.bundle.estimator.predict_proba(applicant_df)[:, 1]
        return float(probabilities[0])

    def predict_one(self, applicant: ApplicantInput) -> PredictionResponse:
        start = time.perf_counter()

        raw_frame = pd.DataFrame([applicant.model_dump()])
        engineered = apply_feature_engineering(raw_frame)

        pd_score = self._predict_probability(engineered)
        risk_band = assign_risk_band(pd_score=pd_score, inference=self.config.inference)
        decision = decision_recommendation(pd_score=pd_score, inference=self.config.inference)
        reason_codes = reason_codes_for_row(
            estimator=self.bundle.estimator,
            applicant_frame=engineered,
            top_k=3,
        )

        request_id = str(uuid.uuid4())
        latency_ms = int((time.perf_counter() - start) * 1000)

        self.prediction_store.put_prediction(
            PredictionRecord(
                request_id=request_id,
                application_id=applicant.application_id,
                model_version=self.bundle.model_version,
                pd_score=pd_score,
                risk_band=risk_band,
                decision=decision,
                reason_codes=reason_codes,
            )
        )

        log_event(
            logger,
            "prediction_completed",
            request_id=request_id,
            model_version=self.bundle.model_version,
            latency_ms=latency_ms,
            status="success",
        )

        return PredictionResponse(
            request_id=request_id,
            model_version=self.bundle.model_version,
            pd_score=pd_score,
            risk_band=risk_band,
            decision_recommendation=decision,
            reason_codes=reason_codes,
            latency_ms=latency_ms,
        )

    def predict_batch(self, applicants: list[ApplicantInput]) -> BatchPredictionResponse:
        batch_id = str(uuid.uuid4())
        prediction_ids: list[str] = []
        batch_predictions: list[dict[str, Any]] = []

        for applicant in applicants:
            prediction = self.predict_one(applicant)
            prediction_ids.append(prediction.request_id)
            batch_predictions.append(prediction.model_dump())

        output_key = f"{self.bundle.model_version}/batch_outputs/{batch_id}.json"
        output_uri = self.artifact_store.put_bytes(
            output_key,
            json.dumps(batch_predictions, indent=2).encode("utf-8"),
        )

        return BatchPredictionResponse(
            batch_id=batch_id,
            request_count=len(applicants),
            output_s3_path=output_uri,
            prediction_ids=prediction_ids,
        )

    def fetch_prediction(self, request_id: str) -> Optional[dict[str, Any]]:
        return self.prediction_store.get_prediction(request_id)


def load_prediction_service(
    base_config_path: Union[str, Path] = "configs/default.yaml",
    env_config_path: Optional[Union[str, Path]] = "configs/local.yaml",
) -> PredictionService:
    config = load_config(base_path=base_config_path, env_path=env_config_path)
    return PredictionService.from_config(config)
