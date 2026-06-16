"""End-to-end scikit-learn pipeline: raw vehicle features -> predicted price.

The whole preprocessing + model is a single sklearn Pipeline so that the
backend only has to load one artifact and call ``.predict(raw_dataframe)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    HIGH_CARD_CATEGORICAL,
    LOW_CARD_CATEGORICAL,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)

# Engineered numeric columns appended by ``engineer_features``.
ENGINEERED = ["age", "age_x_odometer"]
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text columns and add age-based interaction features.

    Module-level (picklable) so it can live inside a FunctionTransformer.
    """
    df = df.copy()

    # text normalisation
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.lower().str.strip()
    if "make" in df.columns:
        df["make"] = df["make"].str.capitalize()

    # feature engineering
    year = pd.to_numeric(df.get("year"), errors="coerce")
    odo = pd.to_numeric(df.get("odometer"), errors="coerce")
    df["age"] = 2026 - year
    df["age_x_odometer"] = (df["age"] * odo) / 1e6
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: impute+scale numeric, one-hot low-card, hash high-card."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    low_card = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = [
        ("num", numeric, ALL_NUMERIC),
        ("low", low_card, LOW_CARD_CATEGORICAL),
    ]

    # High-cardinality columns -> hashing trick (keeps width fixed & small).
    try:
        from category_encoders import HashingEncoder

        high_card = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("hash", HashingEncoder(n_components=32)),
            ]
        )
        transformers.append(("high", high_card, HIGH_CARD_CATEGORICAL))
    except Exception:  # pragma: no cover - fallback if dependency missing
        high_card = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                      max_categories=32)),
            ]
        )
        transformers.append(("high", high_card, HIGH_CARD_CATEGORICAL))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_estimator_and_grid(name: str):
    """Return (estimator, param_grid) for a model name. Compact grids for speed."""
    name = name.lower()
    if name == "ridge":
        return Ridge(random_state=RANDOM_STATE), {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        }
    if name == "random_forest":
        # Bounded depth keeps the artifact small and CI jobs fast.
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), {
            "model__n_estimators": [100],
            "model__max_depth": [15, 20],
            "model__min_samples_leaf": [2],
        }
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE), {
            "model__n_estimators": [100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.1],
        }
    if name == "knn":
        return KNeighborsRegressor(), {
            "model__n_neighbors": [10, 30],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["cosine", "euclidean"],
        }
    raise ValueError(f"Unknown model name: {name!r}")


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> estimator."""
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline(
        steps=[
            ("fe", FunctionTransformer(engineer_features, validate=False)),
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )
