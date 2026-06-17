# Team 1 — Vehicle Price Prediction

## End-to-End Machine Learning Project 2026

**Team:** Team 1  
**Dataset:** [Vehicle Sales Data — Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)  
**Task:** Regression — predict `sellingprice` of used vehicles

---

## Project Description

This project is done by project team 1: Zhilenko Anna, Komarova Diana, Petrichenko Anna.This project builds an end-to-end machine learning pipeline to predict the selling price of used vehicles at auction, based on attributes such as make, model, year, odometer, condition, body type, and transmission.

**Best model:** Random Forest (R² = 0.7996, RMSE ≈ $4,514)

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

| Model                                | RMSE ($) | MAE ($) | R²     |
| ------------------------------------ | -------- | ------- | ------ |
| Random Forest (100 trees, depth=20)  | 4,514    | 2,492   | 0.7996 |
| Gradient Boosting (lr=0.1, depth=5)  | 4,708    | 2,644   | 0.7819 |
| KNN (n=10, distance, cosine)         | 4,957    | 2,526   | 0.7583 |
| Ridge Regression (alpha=1.0)         | 5,566    | 3,148   | 0.6953 |
