"""DynamoDB-compatible persistence abstraction for predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Protocol

import boto3

from credit_risk.schemas.config import AppConfig
from credit_risk.schemas.payloads import PredictionRecord


class PredictionStore(Protocol):
    """Persistence interface for prediction metadata records."""

    def put_prediction(self, record: PredictionRecord) -> None:
        """Persist prediction metadata."""

    def get_prediction(self, request_id: str) -> Optional[dict[str, Any]]:
        """Retrieve stored prediction metadata."""


def _serialize_record(record: PredictionRecord) -> dict[str, Any]:
    payload = record.model_dump()
    payload["timestamp"] = payload["timestamp"].isoformat()
    return payload


def _deserialize_decimal(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_decimal(item) for item in value]
    if isinstance(value, dict):
        return {key: _deserialize_decimal(item) for key, item in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    return value


@dataclass
class LocalPredictionStore:
    """JSONL-backed local storage implementation for predictions."""

    file_path: Path

    def __post_init__(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("", encoding="utf-8")

    def put_prediction(self, record: PredictionRecord) -> None:
        payload = _serialize_record(record)
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def get_prediction(self, request_id: str) -> Optional[dict[str, Any]]:
        for line in self.file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("request_id") == request_id:
                return payload
        return None


@dataclass
class DynamoPredictionStore:
    """DynamoDB-backed storage implementation for predictions."""

    table_name: str
    dynamodb_resource: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.dynamodb_resource is None:
            self.dynamodb_resource = boto3.resource("dynamodb")
        self.table = self.dynamodb_resource.Table(self.table_name)

    def put_prediction(self, record: PredictionRecord) -> None:
        payload = _serialize_record(record)
        self.table.put_item(Item=payload)

    def get_prediction(self, request_id: str) -> Optional[dict[str, Any]]:
        response = self.table.get_item(Key={"request_id": request_id})
        item = response.get("Item")
        if item is None:
            return None
        return _deserialize_decimal(item)


def create_prediction_store(config: AppConfig) -> PredictionStore:
    if config.storage.backend == "local":
        return LocalPredictionStore(Path(config.storage.local_prediction_store_path))

    return DynamoPredictionStore(table_name=config.storage.dynamodb_table)
