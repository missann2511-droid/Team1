"""FastAPI backend for vehicle price prediction (Checkpoint criterion W6.2).

Routes:
    GET  /health         -> liveness + model status
    POST /predict        -> single-sample prediction
    POST /predict_batch  -> multi-sample prediction
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictResponse,
    VehicleFeatures,
)

# --- locate the production model artifact --------------------------------
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "best" / "model_pipeline.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
META_PATH = MODEL_PATH.parent / "metadata.json"

app = FastAPI(title="Vehicle Price Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_model = None
_model_name: Optional[str] = None


def get_model():
    global _model, _model_name
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            try:
                _model_name = json.loads(META_PATH.read_text()).get("model")
            except Exception:
                _model_name = None
    return _model


@app.on_event("startup")
def _warmup() -> None:
    try:
        get_model()
        print(f"[api] Model loaded from {MODEL_PATH} ({_model_name})")
    except Exception as exc:  # noqa: BLE001
        print(f"[api] WARNING: model not loaded at startup: {exc}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = MODEL_PATH.exists()
    if loaded:
        try:
            get_model()  # lazy-load to populate the model name
        except Exception:  # noqa: BLE001
            loaded = False
    return HealthResponse(status="ok", model_loaded=loaded, model_name=_model_name)


def _predict_frame(df: pd.DataFrame):
    model = get_model()
    preds = model.predict(df)
    return [round(float(p), 2) for p in preds]


@app.post("/predict", response_model=PredictResponse)
def predict(features: VehicleFeatures) -> PredictResponse:
    try:
        df = pd.DataFrame([features.model_dump()])
        price = _predict_frame(df)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
    return PredictResponse(predicted_price=price)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")
    try:
        df = pd.DataFrame([i.model_dump() for i in req.items])
        prices = _predict_frame(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}")
    return BatchPredictResponse(predicted_prices=prices, count=len(prices))


@app.get("/")
def root():
    return {"service": "vehicle-price-api", "docs": "/docs", "health": "/health"}
