"""Pydantic schema for model registry records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ModelRegistryRecord(BaseModel):
    """Metadata record for model registration and lineage."""

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_type: Literal["logistic_regression", "random_forest"]
    calibration_method: str = Field(min_length=1)
    artifact_uri: str = Field(min_length=1)
    metrics_uri: str = Field(min_length=1)
    training_data_path: str = Field(min_length=1)
    feature_columns: list[str] = Field(default_factory=list)
    status: Literal["TRAINED", "ACTIVE", "ARCHIVED"] = "TRAINED"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
