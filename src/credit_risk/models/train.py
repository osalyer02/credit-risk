"""Local training entrypoint for credit risk default prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk.config.settings import load_config
from credit_risk.data.clean import clean_training_data
from credit_risk.data.load import load_dataset
from credit_risk.data.validate import validate_training_schema
from credit_risk.features.engineering import apply_feature_engineering
from credit_risk.features.preprocess import build_preprocessor, infer_feature_spec
from credit_risk.models.calibrate import CalibrationResult, calibrate_estimator
from credit_risk.models.evaluate import compute_binary_classification_metrics
from credit_risk.models.explain import global_feature_importance
from credit_risk.schemas.config import AppConfig
from credit_risk.utils.artifacts import ModelBundle, create_artifact_store, save_bundle


@dataclass
class CandidateModelResult:
    name: str
    estimator: Any
    calibration_method: str
    metrics: dict[str, object]
    calibration_candidates: list[dict[str, object]]


@dataclass
class TrainingResult:
    best_model_name: str
    bundle_uri: str
    metrics_uri: str
    metrics: dict[str, object]


def _build_classifier(name: str, config: AppConfig) -> Any:
    if name == "logistic_regression":
        params = config.model.logistic_regression
        return LogisticRegression(
            C=params.C,
            max_iter=params.max_iter,
            class_weight=params.class_weight,
            solver=params.solver,
            random_state=config.split.random_seed,
        )

    if name == "random_forest":
        params = config.model.random_forest
        return RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model candidate: {name}")


def _split_dataset(
    frame: pd.DataFrame,
    target_column: str,
    test_size: float,
    validation_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x = frame.drop(columns=[target_column])
    y = frame[target_column].astype(int)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    val_relative_size = validation_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_relative_size,
        random_state=random_seed,
        stratify=y_train_full,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def _select_best_candidate(candidates: list[CandidateModelResult]) -> CandidateModelResult:
    def score(result: CandidateModelResult) -> tuple[float, float, float]:
        roc_auc = float(result.metrics["roc_auc"])
        brier = float(result.metrics["brier_score"])
        ll = float(result.metrics["log_loss"])
        return (roc_auc, -brier, -ll)

    return max(candidates, key=score)


def _calibration_to_dict(calibration: CalibrationResult) -> dict[str, object]:
    return {
        "method": calibration.method,
        "metrics": calibration.metrics,
    }


def train_from_config(config: AppConfig) -> TrainingResult:
    raw = load_dataset(config.data.training_path)
    validated = validate_training_schema(
        frame=raw,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
    )
    cleaned = clean_training_data(
        frame=validated,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
    )
    featured = apply_feature_engineering(cleaned)

    x_train, x_val, x_test, y_train, y_val, y_test = _split_dataset(
        frame=featured,
        target_column=config.data.target_column,
        test_size=config.split.test_size,
        validation_size=config.split.validation_size,
        random_seed=config.split.random_seed,
    )

    feature_spec = infer_feature_spec(
        frame=featured,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
    )

    thresholds = [0.5, config.inference.approve_threshold, config.inference.decline_threshold]

    candidate_results: list[CandidateModelResult] = []

    for candidate_name in config.model.candidates:
        preprocessor = build_preprocessor(feature_spec)
        classifier = _build_classifier(candidate_name, config)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)

        final_estimator: Any = pipeline
        calibration_method = "none"
        calibration_candidates: list[dict[str, object]] = []

        if config.calibration.enabled:
            best_calibration, all_calibrations = calibrate_estimator(
                estimator=pipeline,
                x_val=x_val,
                y_val=y_val.to_numpy(),
                methods=config.calibration.methods,
            )
            final_estimator = best_calibration.estimator
            calibration_method = best_calibration.method
            calibration_candidates = [_calibration_to_dict(item) for item in all_calibrations]

        test_probabilities = final_estimator.predict_proba(x_test)[:, 1]
        metrics = compute_binary_classification_metrics(
            y_true=y_test.to_numpy(),
            y_prob=np.asarray(test_probabilities),
            thresholds=thresholds,
        )

        candidate_results.append(
            CandidateModelResult(
                name=candidate_name,
                estimator=final_estimator,
                calibration_method=calibration_method,
                metrics=metrics,
                calibration_candidates=calibration_candidates,
            )
        )

    best = _select_best_candidate(candidate_results)
    global_importance = global_feature_importance(
        estimator=best.estimator,
        sample_frame=x_train.head(1000),
    )

    leaderboard = {
        result.name: {
            "metrics": result.metrics,
            "calibration_method": result.calibration_method,
            "calibration_candidates": result.calibration_candidates,
        }
        for result in candidate_results
    }

    final_metrics = {
        "selected_model": best.name,
        "model_version": config.project.model_version,
        "leaderboard": leaderboard,
        "selected_metrics": best.metrics,
    }

    bundle = ModelBundle.create(
        estimator=best.estimator,
        model_name=best.name,
        calibration_method=best.calibration_method,
        feature_columns=[c for c in x_train.columns if c != config.data.id_column],
        target_column=config.data.target_column,
        id_column=config.data.id_column,
        model_version=config.project.model_version,
        metrics=final_metrics,
        global_importance=global_importance,
    )

    artifact_store = create_artifact_store(config)
    bundle_key = f"{config.project.model_version}/{config.artifacts.model_filename}"
    metrics_key = f"{config.project.model_version}/{config.artifacts.metrics_filename}"

    bundle_uri = save_bundle(bundle=bundle, store=artifact_store, key=bundle_key)
    metrics_uri = artifact_store.put_bytes(metrics_key, json.dumps(final_metrics, indent=2).encode("utf-8"))

    # Always persist local copies for local inspection and testing.
    artifact_dir = Path(config.artifacts.local_dir) / config.project.model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / config.artifacts.metrics_filename).write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )

    return TrainingResult(
        best_model_name=best.name,
        bundle_uri=bundle_uri,
        metrics_uri=metrics_uri,
        metrics=final_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk default model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base YAML config")
    parser.add_argument(
        "--env-config",
        default="configs/local.yaml",
        help="Optional environment/local YAML overrides",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(base_path=args.config, env_path=args.env_config)
    result = train_from_config(config)

    print(json.dumps({
        "best_model": result.best_model_name,
        "bundle_uri": result.bundle_uri,
        "metrics_uri": result.metrics_uri,
    }, indent=2))


if __name__ == "__main__":
    main()
