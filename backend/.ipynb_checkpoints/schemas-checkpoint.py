"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class VehicleFeatures(BaseModel):
    """Raw features for a single vehicle (same schema as the training data)."""

    year: int = Field(..., examples=[2014], ge=1900, le=2030)
    make: str = Field(..., examples=["Ford"])
    model: str = Field(..., examples=["Fusion"])
    trim: Optional[str] = Field("unknown", examples=["SE"])
    body: str = Field(..., examples=["Sedan"])
    transmission: str = Field(..., examples=["automatic"])
    state: str = Field(..., examples=["ca"])
    condition: float = Field(..., examples=[35.0], ge=0, le=50)
    odometer: float = Field(..., examples=[42000.0], ge=0)
    color: Optional[str] = Field("unknown", examples=["white"])
    interior: Optional[str] = Field("unknown", examples=["black"])

    model_config = {"extra": "ignore"}


class PredictResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"


class BatchPredictRequest(BaseModel):
    items: List[VehicleFeatures]


class BatchPredictResponse(BaseModel):
    predicted_prices: List[float]
    count: int
    currency: str = "USD"


class HealthResponse(BaseModel):
    # disable pydantic's protected "model_" namespace so these fields are kept
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
