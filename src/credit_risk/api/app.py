"""FastAPI application factory and local run entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI

from credit_risk.api.routes import router
from credit_risk.config.settings import load_config
from credit_risk.scoring.predict import PredictionService


def create_app(
    base_config_path: Union[str, Path] = "configs/default.yaml",
    env_config_path: Optional[Union[str, Path]] = "configs/local.yaml",
) -> FastAPI:
    config = load_config(base_path=base_config_path, env_path=env_config_path)

    app = FastAPI(title=config.api.title)
    app.include_router(router)

    app.state.config = config

    try:
        app.state.prediction_service = PredictionService.from_config(config)
        app.state.startup_error = None
    except Exception as exc:  # pragma: no cover - defensive startup guard
        app.state.prediction_service = None
        app.state.startup_error = str(exc)

    return app


def main() -> None:
    app = create_app()
    config = app.state.config
    uvicorn.run(app, host=config.api.host, port=config.api.port)


app = create_app()


if __name__ == "__main__":
    main()
