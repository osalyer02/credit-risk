"""Credit policy decisioning rules."""

from __future__ import annotations

from typing import Literal

from credit_risk.schemas.config import InferenceConfig


def assign_risk_band(pd_score: float, inference: InferenceConfig) -> str:
    for band in inference.risk_bands:
        if pd_score < band.max_pd:
            return band.name
    return inference.risk_bands[-1].name


def decision_recommendation(
    pd_score: float,
    inference: InferenceConfig,
) -> Literal["APPROVE", "REVIEW", "DECLINE"]:
    if pd_score < inference.approve_threshold:
        return "APPROVE"
    if pd_score < inference.decline_threshold:
        return "REVIEW"
    return "DECLINE"
