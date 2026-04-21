"""Pydantic configuration models for the credit risk platform."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RiskBandConfig(BaseModel):
    """Defines a probability threshold band for risk categorization."""

    name: str = Field(min_length=1)
    max_pd: float = Field(ge=0.0, le=1.0)


class ProjectConfig(BaseModel):
    name: str
    model_version: str


class DataConfig(BaseModel):
    training_path: str
    target_column: str = "default_flag"
    id_column: str = "application_id"


class SplitConfig(BaseModel):
    validation_size: float = Field(gt=0.0, lt=1.0)
    test_size: float = Field(gt=0.0, lt=1.0)
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_total_holdout(self) -> "SplitConfig":
        if (self.validation_size + self.test_size) >= 0.8:
            raise ValueError("Validation size + test size must be < 0.8")
        return self


class LogisticRegressionConfig(BaseModel):
    C: float = Field(gt=0.0)
    max_iter: int = Field(gt=10)
    class_weight: Optional[str] = "balanced"
    solver: str = "liblinear"


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(gt=10)
    max_depth: Optional[int] = Field(default=None, gt=1)
    min_samples_leaf: int = Field(default=1, ge=1)
    random_state: int = 42


class ModelConfig(BaseModel):
    candidates: list[Literal["logistic_regression", "random_forest"]] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest"]
    )
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)


class CalibrationConfig(BaseModel):
    enabled: bool = True
    methods: list[Literal["sigmoid", "isotonic"]] = Field(default_factory=lambda: ["sigmoid"])

    @field_validator("methods")
    @classmethod
    def ensure_non_empty(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("At least one calibration method is required")
        return values


class InferenceConfig(BaseModel):
    approve_threshold: float = Field(ge=0.0, le=1.0)
    decline_threshold: float = Field(ge=0.0, le=1.0)
    risk_bands: list[RiskBandConfig]

    @model_validator(mode="after")
    def validate_thresholds(self) -> "InferenceConfig":
        if self.approve_threshold >= self.decline_threshold:
            raise ValueError("approve_threshold must be less than decline_threshold")

        if not self.risk_bands:
            raise ValueError("At least one risk band is required")

        previous = 0.0
        for band in self.risk_bands:
            if band.max_pd < previous:
                raise ValueError("Risk band max_pd values must be non-decreasing")
            previous = band.max_pd

        if self.risk_bands[-1].max_pd < 1.0:
            raise ValueError("Last risk band max_pd must be 1.0")

        return self


class ArtifactsConfig(BaseModel):
    local_dir: str = "artifacts/models"
    metrics_dir: str = "artifacts/metrics"
    model_filename: str = "model_bundle.joblib"
    metrics_filename: str = "metrics.json"
    validation_report_filename: str = "validation_report.json"
    registry_filename: str = "model_registry_record.json"


class StorageConfig(BaseModel):
    local_prediction_store_path: str = "artifacts/predictions.jsonl"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Credit Risk Platform API"
    debug: bool = False


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="ignore")

    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    calibration: CalibrationConfig
    inference: InferenceConfig
    artifacts: ArtifactsConfig
    storage: StorageConfig
    api: APIConfig
