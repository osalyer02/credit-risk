"""AWS Lambda handler wrapper for the FastAPI application."""

from __future__ import annotations

from credit_risk.api.app import app

try:
    from mangum import Mangum
except ImportError as exc:  # pragma: no cover - only relevant in AWS runtime packaging
    raise RuntimeError("mangum is required for lambda_handler") from exc

handler = Mangum(app)
