from __future__ import annotations

from credit_risk.config.settings import load_config
from credit_risk.scoring.rules import assign_risk_band, decision_recommendation


def test_decision_rules_apply_thresholds():
    config = load_config(base_path="configs/default.yaml", env_path="configs/local.yaml")

    assert decision_recommendation(0.04, config.inference) == "APPROVE"
    assert decision_recommendation(0.10, config.inference) == "REVIEW"
    assert decision_recommendation(0.30, config.inference) == "DECLINE"


def test_risk_band_assignment_is_config_driven():
    config = load_config(base_path="configs/default.yaml", env_path="configs/local.yaml")

    assert assign_risk_band(0.03, config.inference) == "A"
    assert assign_risk_band(0.09, config.inference) == "B"
    assert assign_risk_band(0.50, config.inference) == "E"
