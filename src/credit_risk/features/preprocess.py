"""Preprocessing pipeline builders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]


def infer_feature_spec(frame: pd.DataFrame, target_column: str, id_column: str) -> FeatureSpec:
    """Infer numeric/categorical feature columns from a dataframe."""

    candidate_columns = [c for c in frame.columns if c not in {target_column, id_column}]

    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in candidate_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return FeatureSpec(numeric_features=numeric_features, categorical_features=categorical_features)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build a reusable preprocessing graph for tabular data."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_features:
        transformers.append(("num", numeric_transformer, spec.numeric_features))
    if spec.categorical_features:
        transformers.append(("cat", categorical_transformer, spec.categorical_features))

    if not transformers:
        raise ValueError("No features available to build preprocessing pipeline")

    return ColumnTransformer(transformers=transformers, remainder="drop")
