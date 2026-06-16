.PHONY: install data train select api ui compose test pipeline

install:
	pip install -r backend/requirements.txt -r frontend/requirements.txt

data:
	python -m src.preprocess

train:
	python -m src.train --model ridge
	python -m src.train --model random_forest
	python -m src.train --model gradient_boosting
	python -m src.train --model knn

select:
	python -m src.select_best

pipeline: data train select

api:
	uvicorn backend.main:app --reload --port 8000

ui:
	BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py

compose:
	docker compose up --build

test:
	pytest -q
