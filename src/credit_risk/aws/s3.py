"""S3-compatible artifact persistence abstractions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Protocol

import boto3


class ArtifactStore(Protocol):
    """Common interface for artifact persistence backends."""

    def put_bytes(self, key: str, data: bytes) -> str:
        """Store bytes and return a fully qualified location."""

    def get_bytes(self, key: str) -> bytes:
        """Load bytes at key."""


@dataclass
class LocalArtifactStore:
    """File-system implementation of artifact persistence."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = self.root_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.write_bytes(data)
        return str(path)

    def get_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        return path.read_bytes()

    def put_json(self, key: str, payload: dict[str, Any]) -> str:
        return self.put_bytes(key, json.dumps(payload, indent=2).encode("utf-8"))


@dataclass
class S3ArtifactStore:
    """Amazon S3-backed implementation of artifact persistence."""

    bucket: str
    prefix: str = ""
    s3_client: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.s3_client is None:
            self.s3_client = boto3.client("s3")

    def _full_key(self, key: str) -> str:
        if not self.prefix:
            return key
        return f"{self.prefix.rstrip('/')}/{key.lstrip('/')}"

    def put_bytes(self, key: str, data: bytes) -> str:
        full_key = self._full_key(key)
        self.s3_client.put_object(Bucket=self.bucket, Key=full_key, Body=data)
        return f"s3://{self.bucket}/{full_key}"

    def get_bytes(self, key: str) -> bytes:
        full_key = self._full_key(key)
        response = self.s3_client.get_object(Bucket=self.bucket, Key=full_key)
        body = response["Body"].read()
        return bytes(body)

    def put_json(self, key: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        return self.put_bytes(key, encoded)


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def parse_s3_uri(path: str) -> tuple[str, str]:
    without_prefix = path.replace("s3://", "", 1)
    bucket, _, key = without_prefix.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {path}")
    return bucket, key


def read_s3_bytes(path: str, s3_client: Optional[Any] = None) -> bytes:
    bucket, key = parse_s3_uri(path)
    client = s3_client or boto3.client("s3")
    response = client.get_object(Bucket=bucket, Key=key)
    return bytes(response["Body"].read())


def read_s3_dataframe(path: str, format_hint: str, s3_client: Optional[Any] = None) -> Any:
    import pandas as pd

    payload = read_s3_bytes(path=path, s3_client=s3_client)
    buffer = BytesIO(payload)
    if format_hint == "csv":
        return pd.read_csv(buffer)
    if format_hint == "parquet":
        return pd.read_parquet(buffer)
    raise ValueError(f"Unsupported format hint: {format_hint}")
