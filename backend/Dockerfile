# Backend image (Checkpoint criterion W7.1)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code. `src` is required because the saved pipeline references
# src.pipeline.engineer_features and must be importable for unpickling.
COPY src ./src
COPY backend ./backend
COPY models ./models

EXPOSE 8000

# Model path can be overridden at runtime via MODEL_PATH
ENV MODEL_PATH=/app/models/best/model_pipeline.joblib

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
