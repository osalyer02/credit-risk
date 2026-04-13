"""Model artifact serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib

from credit_risk.aws.s3 import ArtifactStore, LocalArtifactStore
from credit_risk.schemas.config import AppConfig


@dataclass
class ModelBundle:
    """Serializable package for model inference and metadata."""

    estimator: Any
    model_name: str
    calibration_method: str
    feature_columns: list[str]
    target_column: str
    id_column: str
    model_version: str
    metrics: dict[str, Any]
    global_importance: list[dict[str, Any]]
    trained_at_utc: str

    @classmethod
    def create(
        cls,
        estimator: Any,
        model_name: str,
        calibration_method: str,
        feature_columns: list[str],
        target_column: str,
        id_column: str,
        model_version: str,
        metrics: dict[str, Any],
        global_importance: list[dict[str, Any]],
    ) -> "ModelBundle":
        return cls(
            estimator=estimator,
            model_name=model_name,
            calibration_method=calibration_method,
            feature_columns=feature_columns,
            target_column=target_column,
            id_column=id_column,
            model_version=model_version,
            metrics=metrics,
            global_importance=global_importance,
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
        )


def create_artifact_store(config: AppConfig) -> ArtifactStore:
    if config.storage.backend == "local":
        return LocalArtifactStore(Path(config.artifacts.local_dir))

    from credit_risk.aws.s3 import S3ArtifactStore

    if not config.storage.s3_bucket:
        raise ValueError("storage.s3_bucket is required for aws backend")

    return S3ArtifactStore(bucket=config.storage.s3_bucket, prefix=config.storage.s3_prefix)


def save_bundle(bundle: ModelBundle, store: ArtifactStore, key: str) -> str:
    with NamedTemporaryFile(suffix=".joblib") as handle:
        joblib.dump(bundle, Path(handle.name))
        data = Path(handle.name).read_bytes()
    return store.put_bytes(key, data)


def load_bundle(store: ArtifactStore, key: str) -> ModelBundle:
    data = store.get_bytes(key)
    with NamedTemporaryFile(suffix=".joblib") as handle:
        path = Path(handle.name)
        path.write_bytes(data)
        bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Artifact does not contain a ModelBundle")
    return bundle


def save_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
