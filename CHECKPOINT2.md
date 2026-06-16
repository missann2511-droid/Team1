# Checkpoint 2 — Pipeline, Serving & Deployment

Team 1 — Zhilenko Anna, Komarova Diana, Petrichenko Anna
Dataset: Vehicle Sales Data (Kaggle) · Task: regression on `sellingprice`

This checkpoint turns the Checkpoint-1 notebooks into a reproducible pipeline,
a serving backend, an interactive frontend, containers, CI, and a cloud
deployment path. Criterion references (W6.1 … W8.2) match the grading sheet.

---

## What maps to which criterion

| Criterion | Deliverable |
|-----------|-------------|
| **W6.1** Data preprocessing pipeline | `src/download_data.py` → `src/preprocess.py` (download **before** processing; outputs to `data/processed/`; split-before-fit avoids leakage) |
| **W6.1** Model training pipeline (parallel) | `src/train.py` trains one model per invocation; `.github/workflows/ci.yml` runs all four as a **matrix of parallel jobs**; `src/select_best.py` publishes the winner |
| **W6.2** Backend `/health` | `GET /health` in `backend/main.py` |
| **W6.2** Backend `/predict` | `POST /predict` (single sample) |
| **W6.2** Backend `/predict_batch` | `POST /predict_batch` (list of samples) |
| **W6.2** Frontend single prediction | `frontend/app.py` → "Single prediction" tab |
| **W6.2** Frontend batched prediction | `frontend/app.py` → "Batch prediction" tab (CSV upload) |
| **W7.1** Backend docker image | `backend/Dockerfile` |
| **W7.1** Frontend docker image | `frontend/Dockerfile` |
| **W7.2** Docker compose | `docker-compose.yml` |
| **W7.2** GitHub Actions | `.github/workflows/ci.yml` |
| **W8.1** Deploy on Y.Cloud | `deploy/DEPLOY_YCLOUD.md` |
| **W8.2** Autodeploy with GH Actions | `.github/workflows/deploy.yml` |

---

## Architecture

```
            ┌──────────────┐      HTTP/JSON      ┌──────────────┐
 user  ───▶ │  Streamlit   │ ──────────────────▶ │   FastAPI    │
            │  frontend    │   /predict[_batch]  │   backend    │
            │  :8501       │ ◀────────────────── │   :8000      │
            └──────────────┘                     └──────┬───────┘
                                                        │ joblib.load
                                                 models/best/model_pipeline.joblib
                                                 (FE → preprocess → estimator)
```

The model artifact is a **single end-to-end scikit-learn `Pipeline`**
(feature engineering → `ColumnTransformer` → estimator), so the backend just
calls `pipeline.predict(raw_dataframe)` — no preprocessing duplicated in serving.

---

## Run locally

```bash
make install        # backend + frontend deps
make pipeline       # download → preprocess → train 4 models → select best
make api            # http://localhost:8000/docs
make ui             # http://localhost:8501   (separate terminal)
```

Or everything in containers:

```bash
make pipeline       # produce models/best/model_pipeline.joblib first
docker compose up --build
# frontend: http://localhost:8501   backend: http://localhost:8000/health
```

## API examples

```bash
curl localhost:8000/health

curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "year":2014,"make":"Ford","model":"Fusion","trim":"SE","body":"Sedan",
  "transmission":"automatic","state":"ca","condition":35,"odometer":42000,
  "color":"white","interior":"black"}'

curl -X POST localhost:8000/predict_batch -H 'content-type: application/json' -d '{
  "items":[{"year":2014,"make":"Ford","model":"Fusion","trim":"SE","body":"Sedan",
  "transmission":"automatic","state":"ca","condition":35,"odometer":42000,
  "color":"white","interior":"black"}]}'
```

## Notes on grids

GridSearch grids in `src/pipeline.py` are kept compact so the four parallel CI
jobs and cloud builds finish quickly. Widen them (e.g. add `max_depth=None`,
more `n_estimators`) for a final tuned run — the pipeline and serving code are
unchanged.

## Fast local runs

Training on the full sample with deep trees is slow on a laptop. Cap the rows
for a quick smoke run (CI leaves it unset → full data):

```bash
MAX_TRAIN_ROWS=8000 make train && make select
```

## Tests

`tests/test_api.py` smoke-tests `/health`, `/predict`, `/predict_batch` with
`pytest` (also runnable in CI).
