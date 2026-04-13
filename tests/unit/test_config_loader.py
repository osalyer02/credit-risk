from __future__ import annotations

import yaml

from credit_risk.config.settings import load_config


def test_load_config_supports_yaml_and_env_overrides(tmp_path, monkeypatch):
    base = {
        "project": {"name": "credit-risk-platform", "model_version": "base"},
        "data": {
            "training_path": "data/raw/data.csv",
            "target_column": "default_flag",
            "id_column": "application_id",
        },
        "split": {"validation_size": 0.2, "test_size": 0.2, "random_seed": 42},
        "model": {
            "candidates": ["logistic_regression", "random_forest"],
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 500,
                "class_weight": "balanced",
                "solver": "liblinear",
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_leaf": 10,
                "random_state": 42,
            },
        },
        "calibration": {"enabled": True, "methods": ["sigmoid"]},
        "inference": {
            "approve_threshold": 0.05,
            "decline_threshold": 0.2,
            "risk_bands": [
                {"name": "A", "max_pd": 0.1},
                {"name": "E", "max_pd": 1.0},
            ],
        },
        "artifacts": {
            "local_dir": "artifacts",
            "model_filename": "bundle.joblib",
            "metrics_filename": "metrics.json",
        },
        "storage": {
            "backend": "local",
            "s3_bucket": None,
            "s3_prefix": "models/",
            "dynamodb_table": "credit_predictions",
            "local_prediction_store_path": "artifacts/predictions.jsonl",
        },
        "api": {"host": "0.0.0.0", "port": 8000, "title": "API", "debug": False},
    }

    env_override = {
        "project": {"model_version": "env-file"},
        "api": {"debug": True},
    }

    base_path = tmp_path / "base.yaml"
    env_path = tmp_path / "env.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    env_path.write_text(yaml.safe_dump(env_override), encoding="utf-8")

    monkeypatch.setenv("CRP_PROJECT__MODEL_VERSION", "env-var")

    config = load_config(base_path=base_path, env_path=env_path)

    assert config.project.model_version == "env-var"
    assert config.api.debug is True
