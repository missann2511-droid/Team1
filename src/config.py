"""Central configuration: paths, dataset URL, feature lists, model registry."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "car_prices_sample.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
BEST_DIR = MODELS_DIR / "best"
REPORTS_DIR = ROOT / "reports"

for _d in (PROCESSED_DIR, BEST_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset -------------------------------------------------------------
# Public mirror of the team sample. Override with DATA_URL env var if needed.
DATA_URL = os.getenv(
    "DATA_URL",
    "https://raw.githubusercontent.com/syedanwarafridi/vehicle-sales-data/main/car_prices.csv",
)

TARGET = "sellingprice"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Feature schema (raw input expected from the user / API) -------------
NUMERIC_FEATURES = ["year", "condition", "odometer"]
LOW_CARD_CATEGORICAL = ["make", "body", "transmission", "state", "color", "interior"]
HIGH_CARD_CATEGORICAL = ["model", "trim"]
DROP_COLUMNS = ["vin", "seller", "saledate"]

# All raw features the API accepts for one prediction
RAW_FEATURES = NUMERIC_FEATURES + LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL

# --- Model registry: name -> (estimator factory, param grid) -------------
# Grids are intentionally compact so the parallel CI jobs finish quickly.
MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting", "knn"]
