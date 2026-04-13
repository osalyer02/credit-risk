"""Dataset ingestion helpers for local files and S3 objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from credit_risk.aws.s3 import is_s3_uri, read_s3_dataframe


class DataLoadError(RuntimeError):
    """Raised when dataset loading fails."""


def detect_format(path: str) -> str:
    if path.endswith(".csv"):
        return "csv"
    if path.endswith(".parquet"):
        return "parquet"
    raise DataLoadError(f"Unsupported file extension for data path: {path}")


def load_dataset(path: str, s3_client: Optional[Any] = None) -> pd.DataFrame:
    """Load CSV or Parquet data from local path or S3 URI."""

    fmt = detect_format(path)

    if is_s3_uri(path):
        return read_s3_dataframe(path=path, format_hint=fmt, s3_client=s3_client)

    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError(f"Data file not found: {file_path}")

    if fmt == "csv":
        return pd.read_csv(file_path)
    if fmt == "parquet":
        return pd.read_parquet(file_path)

    raise DataLoadError(f"Unsupported format: {fmt}")
