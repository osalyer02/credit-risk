from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from credit_risk.api.app import create_app
from credit_risk.models.train import train_from_config


def _sample_applicant() -> dict[str, object]:
    return {
        "application_id": "app-live-001",
        "annual_income": 92000,
        "loan_amount": 18000,
        "dti": 16.5,
        "fico_range_low": 690,
        "fico_range_high": 695,
        "revolving_utilization": 38.0,
        "open_accounts": 9,
        "delinquencies_2y": 0,
    }


def test_api_predict_and_retrieve(
    local_test_config,
    synthetic_training_frame,
    write_config_file,
):
    data_path = Path(local_test_config.artifacts.local_dir).parent / "train.csv"
    synthetic_training_frame.to_csv(data_path, index=False)

    config_payload = local_test_config.model_dump(mode="python")
    config_payload["data"]["training_path"] = str(data_path)
    config = local_test_config.__class__.model_validate(config_payload)
    train_from_config(config)

    config_path = write_config_file(config)
    app = create_app(base_config_path=config_path, env_config_path=None)

    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        predict_resp = client.post("/predict", json=_sample_applicant())
        assert predict_resp.status_code == 200
        payload = predict_resp.json()
        assert 0 <= payload["pd_score"] <= 1
        assert payload["decision_recommendation"] in {"APPROVE", "REVIEW", "DECLINE"}
        request_id = payload["request_id"]

        fetch_resp = client.get(f"/prediction/{request_id}")
        assert fetch_resp.status_code == 200
        assert fetch_resp.json()["request_id"] == request_id

        batch_resp = client.post("/predict_batch", json={"applicants": [_sample_applicant()]})
        assert batch_resp.status_code == 200
        assert batch_resp.json()["request_count"] == 1
