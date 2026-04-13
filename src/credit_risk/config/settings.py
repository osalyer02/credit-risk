"""Application configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from credit_risk.schemas.config import AppConfig

ENV_PREFIX = "CRP_"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping at root: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_env_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _env_overrides(prefix: str = ENV_PREFIX) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    for key, raw_value in os.environ.items():
        if not key.startswith(prefix):
            continue

        path_parts = key[len(prefix) :].lower().split("__")
        cursor = overrides
        for part in path_parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path_parts[-1]] = _coerce_env_value(raw_value)

    return overrides


def load_config(
    base_path: Union[str, Path] = "configs/default.yaml",
    env_path: Optional[Union[str, Path]] = "configs/local.yaml",
) -> AppConfig:
    """Load YAML config files with optional environment-variable overrides."""

    base_dict = _read_yaml(Path(base_path))

    if env_path is not None and Path(env_path).exists():
        env_dict = _read_yaml(Path(env_path))
        merged = _deep_merge(base_dict, env_dict)
    else:
        merged = base_dict

    merged = _deep_merge(merged, _env_overrides())
    return AppConfig.model_validate(merged)
