"""Streamlit frontend (Checkpoint criterion W6.2).

Two modes:
  * Single prediction  -> form -> POST /predict
  * Batch prediction    -> CSV upload -> POST /predict_batch
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.caption("E2E ML Project · Team 1 — Zhilenko, Komarova, Petrichenko")

# --- backend health ------------------------------------------------------
with st.sidebar:
    st.header("Backend")
    st.write(f"`{BACKEND_URL}`")
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        if h.get("model_loaded"):
            st.success(f"Healthy · model: {h.get('model_name') or 'loaded'}")
        else:
            st.warning("API up, model not loaded")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Backend unreachable: {exc}")

REQUIRED = ["year", "make", "model", "trim", "body", "transmission",
            "state", "condition", "odometer", "color", "interior"]

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# --- single --------------------------------------------------------------
with tab_single:
    st.subheader("Predict one vehicle")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", 1990, 2026, 2014)
        make = st.text_input("Make", "Ford")
        model_ = st.text_input("Model", "Fusion")
        trim = st.text_input("Trim", "SE")
        body = st.text_input("Body", "Sedan")
        transmission = st.selectbox("Transmission", ["automatic", "manual"])
    with c2:
        state = st.text_input("State", "ca")
        condition = st.slider("Condition (1–50)", 1.0, 50.0, 35.0)
        odometer = st.number_input("Odometer (miles)", 0.0, 500000.0, 42000.0, step=1000.0)
        color = st.text_input("Color", "white")
        interior = st.text_input("Interior", "black")

    if st.button("Predict price", type="primary"):
        payload = {
            "year": year, "make": make, "model": model_, "trim": trim, "body": body,
            "transmission": transmission, "state": state, "condition": condition,
            "odometer": odometer, "color": color, "interior": interior,
        }
        try:
            r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
            r.raise_for_status()
            price = r.json()["predicted_price"]
            st.metric("Predicted selling price", f"${price:,.0f}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")

# --- batch ---------------------------------------------------------------
with tab_batch:
    st.subheader("Predict from a CSV file")
    st.write("CSV must contain columns: " + ", ".join(f"`{c}`" for c in REQUIRED))
    sample = pd.DataFrame([
        {"year": 2014, "make": "Ford", "model": "Fusion", "trim": "SE", "body": "Sedan",
         "transmission": "automatic", "state": "ca", "condition": 35, "odometer": 42000,
         "color": "white", "interior": "black"},
        {"year": 2012, "make": "BMW", "model": "3 Series", "trim": "328i", "body": "Sedan",
         "transmission": "automatic", "state": "fl", "condition": 28, "odometer": 76000,
         "color": "black", "interior": "black"},
    ])
    st.download_button("Download sample CSV", sample.to_csv(index=False),
                       "sample_vehicles.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.write("Preview:", df.head())
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif st.button("Predict batch", type="primary"):
            items = df[REQUIRED].to_dict(orient="records")
            try:
                r = requests.post(f"{BACKEND_URL}/predict_batch",
                                  json={"items": items}, timeout=120)
                r.raise_for_status()
                preds = r.json()["predicted_prices"]
                out = df.copy()
                out["predicted_price"] = preds
                st.success(f"Predicted {len(preds)} vehicles.")
                st.dataframe(out)
                st.download_button("Download results CSV", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Batch request failed: {exc}")
