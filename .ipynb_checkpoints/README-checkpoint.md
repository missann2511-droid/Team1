# Team 1 — Vehicle Price Prediction

## End-to-End Machine Learning Project 2026

**Team:** Team 1  
**Dataset:** [Vehicle Sales Data — Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)  
**Task:** Regression — predict `sellingprice` of used vehicles

---

## Project Description

This project is done by project team 1: Zhilenko Anna, Komarova Diana, Petrichenko Anna.This project builds an end-to-end machine learning pipeline to predict the selling price of used vehicles at auction, based on attributes such as make, model, year, odometer, condition, body type, and transmission.

**Best model:** K-Nearest Neighbors (R² = 0.8682, RMSE ≈ $3,091)

---

## Repository Structure

```
├── data/
│   ├── car_prices_sample.csv       # Raw dataset (team sample)
│   └── processed/                  # Preprocessed train/test splits + models
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis (Week 2-3)
│   ├── DataPreprocessing.ipynb     # Feature Engineering & Encoding (Week 3)
│   ├── ModelTraining.ipynb         # Model training & GridSearchCV (Week 4)
│   └── ModelDeploymentPrep.ipynb   # Giskard scan & model saving (Week 5)
├── reports/
│   ├── ModelTrainingReport.html    # Full model evaluation report
│   ├── giskard_scan_report.html    # Giskard vulnerability scan
│   └── *.png                       # EDA and model plots
├── backend/                        # FastAPI backend (Checkpoint 2)
├── frontend/                       # Streamlit / React frontend (Checkpoint 2)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

```bash
pip install -r requirements.txt
# Then run notebooks in order: EDA → DataPreprocessing → ModelTraining → ModelDeploymentPrep
```

## Results Summary

| Model | RMSE ($) | MAE ($) | R² |
|-------|---------|---------|-----|
| KNN (n=10, distance, cosine) | 3,091 | 2,012 | 0.8682 |
| Random Forest (100 trees, depth=20) | 3,138 | 2,116 | 0.8642 |
| Gradient Boosting (lr=0.1, depth=5) | 3,248 | 2,251 | 0.8545 |
| Ridge Regression (alpha=1.0) | 4,058 | 2,952 | 0.7729 |
