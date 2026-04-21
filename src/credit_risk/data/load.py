"""Dataset ingestion helpers for local files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoadError(RuntimeError):
    """Raised when dataset loading fails."""


def detect_format(path: str) -> str:
    if path.endswith(".csv"):
        return "csv"
    if path.endswith(".parquet"):
        return "parquet"
    raise DataLoadError(f"Unsupported file extension for data path: {path}")


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV or Parquet data from a local file path."""

    fmt = detect_format(path)

    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError(f"Data file not found: {file_path}")

    if fmt == "csv":
        return pd.read_csv(file_path)
    if fmt == "parquet":
        return pd.read_parquet(file_path)

    raise DataLoadError(f"Unsupported format: {fmt}")
