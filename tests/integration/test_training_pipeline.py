from __future__ import annotations

from pathlib import Path

from credit_risk.models.train import train_from_config


def test_training_pipeline_runs_end_to_end(local_test_config, synthetic_training_frame):
    data_path = Path(local_test_config.artifacts.local_dir).parent / "train.csv"
    synthetic_training_frame.to_csv(data_path, index=False)

    config_payload = local_test_config.model_dump(mode="python")
    config_payload["data"]["training_path"] = str(data_path)
    config = local_test_config.__class__.model_validate(config_payload)

    result = train_from_config(config)

    assert result.best_model_name in {"logistic_regression", "random_forest"}
    assert "selected_metrics" in result.metrics
    assert Path(result.bundle_uri).exists()
    assert Path(result.metrics_uri).exists()
    assert "artifacts/models" in str(result.bundle_uri)
    assert "artifacts/metrics" in str(result.metrics_uri)

    metrics_dir = Path(local_test_config.artifacts.metrics_dir) / local_test_config.project.model_version
    assert (metrics_dir / local_test_config.artifacts.validation_report_filename).exists()
    assert (metrics_dir / local_test_config.artifacts.registry_filename).exists()
