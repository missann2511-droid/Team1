"""Train ONE model (Checkpoint criterion W6.1).

Each model is trained by an independent invocation so the CI pipeline can run
them as parallel jobs (matrix strategy). Usage:

    python -m src.train --model knn

Outputs:
    models/<model>_pipeline.joblib   full fitted pipeline
    data/processed/<model>_results.json   test metrics + best params
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV

from .config import MODELS_DIR, MODEL_NAMES, PROCESSED_DIR, RAW_FEATURES, TARGET
from .pipeline import build_pipeline, get_estimator_and_grid


def load_split():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    # Optional row cap for fast local/dev runs. CI leaves this unset -> full data.
    cap = os.getenv("MAX_TRAIN_ROWS")
    if cap:
        n = int(cap)
        train = train.sample(min(n, len(train)), random_state=42).reset_index(drop=True)
        print(f"[train] MAX_TRAIN_ROWS set -> using {len(train)} train rows")
    return (
        train[RAW_FEATURES], train[TARGET],
        test[RAW_FEATURES], test[TARGET],
    )


def train_one(name: str, cv: int = 3) -> dict:
    X_train, y_train, X_test, y_test = load_split()
    estimator, grid = get_estimator_and_grid(name)
    pipe = build_pipeline(estimator)

    print(f"[train:{name}] GridSearchCV over {grid}")
    t0 = time.time()
    gs = GridSearchCV(
        pipe, grid, scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    pred = best.predict(X_test)
    result = {
        "model": name,
        "rmse": float(root_mean_squared_error(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "cv_rmse": float(-gs.best_score_),
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        "train_seconds": round(time.time() - t0, 1),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODELS_DIR / f"{name}_pipeline.joblib")
    with open(PROCESSED_DIR / f"{name}_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[train:{name}] RMSE=${result['rmse']:,.2f}  MAE=${result['mae']:,.2f}  "
          f"R2={result['r2']:.4f}  ({result['train_seconds']}s)")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()
    train_one(args.model, cv=args.cv)
