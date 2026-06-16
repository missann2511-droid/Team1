"""Preprocessing pipeline (Checkpoint criterion W6.1).

Design avoids data leakage: row-level cleanup + train/test split happen here,
but ALL fitted transforms (imputers, scaler, encoders) are fit inside the
model pipeline on the TRAIN split only (see src/pipeline.py). The processed
train/test splits are stored under data/processed/ as required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DROP_COLUMNS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_FEATURES,
    RAW_PATH,
    TARGET,
    TEST_SIZE,
)
from .download_data import download


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleanup that is safe to do before splitting (no fitted stats)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # drop structurally unusable rows
    df = df.dropna(subset=[TARGET])
    df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()]
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET] > 0]

    # known data-entry error: 'sedan' appearing in the transmission column
    if "transmission" in df.columns:
        df = df[df["transmission"].astype(str).str.lower() != "sedan"]

    # drop non-informative identifier columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # keep only known raw features (+ target); add any missing as NaN
    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    df = df[RAW_FEATURES + [TARGET]].reset_index(drop=True)
    return df


def run() -> None:
    download()
    print(f"[preprocess] Loading {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH, on_bad_lines="skip")
    print(f"[preprocess] Raw shape: {raw.shape}")

    df = clean_rows(raw)
    print(f"[preprocess] After row cleanup: {df.shape}")

    X = df[RAW_FEATURES]
    y = df[TARGET]

    # cap extreme price outliers using TRAIN statistics only -> no leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    p99 = np.percentile(y_train, 99)
    keep = y_train <= p99
    X_train, y_train = X_train[keep], y_train[keep]
    print(f"[preprocess] Train price cap at 99th pct = ${p99:,.0f} "
          f"({(~keep).sum()} train rows dropped)")

    train = X_train.copy(); train[TARGET] = y_train.values
    test = X_test.copy(); test[TARGET] = y_test.values

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[preprocess] Saved train {train.shape} and test {test.shape} "
          f"to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()
