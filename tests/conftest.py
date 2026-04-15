"""Shared pytest fixtures for credit risk platform tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from credit_risk.config.settings import load_config
from credit_risk.schemas.config import AppConfig


@pytest.fixture
def synthetic_training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500

    annual_income = rng.normal(85000, 22000, n).clip(20000, 200000)
    loan_amount = rng.normal(15000, 7000, n).clip(1000, 60000)
    dti = rng.normal(17, 8, n).clip(0, 45)
    fico_low = rng.integers(580, 760, size=n)
    fico_high = fico_low + rng.integers(0, 20, size=n)
    utilization = rng.normal(45, 25, n).clip(0, 150)
    open_accounts = rng.integers(1, 20, size=n)
    delinquencies = rng.integers(0, 4, size=n)

    logits = (
        0.00006 * loan_amount
        - 0.000015 * annual_income
        + 0.03 * dti
        - 0.01 * ((fico_low + fico_high) / 2)
        + 0.02 * utilization
        + 0.35 * delinquencies
        - 0.03 * open_accounts
        + 4.5
    )
    probs = 1 / (1 + np.exp(-logits))
    default_flag = (rng.random(n) < probs).astype(int)

    return pd.DataFrame(
        {
            "application_id": [f"app-{i:04d}" for i in range(n)],
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "dti": dti,
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "revolving_utilization": utilization,
            "open_accounts": open_accounts,
            "delinquencies_2y": delinquencies,
            "default_flag": default_flag,
        }
    )


@pytest.fixture
def local_test_config(tmp_path: Path) -> AppConfig:
    base = load_config(base_path="configs/default.yaml", env_path="configs/local.yaml").model_dump()

    artifact_root = tmp_path / "artifacts"
    artifact_models_dir = artifact_root / "models"
    artifact_metrics_dir = artifact_root / "metrics"
    artifact_models_dir.mkdir(parents=True, exist_ok=True)
    artifact_metrics_dir.mkdir(parents=True, exist_ok=True)
    base["project"]["model_version"] = "test-v1"
    base["artifacts"]["local_dir"] = str(artifact_models_dir)
    base["artifacts"]["metrics_dir"] = str(artifact_metrics_dir)
    base["storage"]["backend"] = "local"
    base["storage"]["local_prediction_store_path"] = str(artifact_root / "predictions.jsonl")

    return AppConfig.model_validate(base)


@pytest.fixture
def write_config_file(tmp_path: Path):
    def _write(config: AppConfig) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False), encoding="utf-8")
        return path

    return _write
