# Как залить минимальные правки на GitHub

Цель: ноутбук, отчёт и README показывают одни и те же **реальные** числа,
а сохранённая модель (Random Forest) им соответствует. Лучшая модель — **Random Forest, R² = 0.7996**.

## Файлы для замены (загрузить поверх старых)

Через `Add file → Upload files` на странице репозитория. Можно перетащить
сразу всё дерево из папки `github_fixed/` — GitHub сохранит пути.

| Файл в репозитории | Откуда взять |
|---|---|
| `notebooks/ModelTraining.ipynb` | `github_fixed/notebooks/ModelTraining.ipynb` |
| `README.md` | `github_fixed/README.md` |
| `reports/ModelTrainingReport.html` | `github_fixed/reports/ModelTrainingReport.html` |
| `ModelTrainingReport.html` (в корне) | `github_fixed/ModelTrainingReport.html` |
| `reports/model1_ridge_plots.png` | `github_fixed/reports/...` |
| `reports/model2_rf_plots.png` | то же |
| `reports/model3_gb_plots.png` | то же |
| `reports/model4_knn_plots.png` | то же |
| `reports/model_comparison.png` | то же |
| `data/processed/ridge_results.json` | `github_fixed/data/processed/...` |
| `data/processed/rf_results.json` | то же |
| `data/processed/gb_results.json` | то же |
| `data/processed/knn_results.json` | то же |
| `data/processed/model_results.json` | то же |

## Что НЕ обязательно трогать

- `models/best/best_model.joblib` — там уже Random Forest. Можно оставить как есть.
  Если хочешь точную копию (depth=20, R²=0.7996) — загрузи версию из
  `github_fixed/models/best/best_model.joblib` (21 МБ, в пределах лимита GitHub).
  При запуске исправленного ноутбука этот файл всё равно пересоздаётся автоматически.

## Сообщение коммита

```
Rebuild ModelTraining: reproducible metrics, Random Forest as best model
```

## Проверка после заливки

Открой `reports/ModelTrainingReport.html` на GitHub — внизу должно быть
«Best model: Random Forest». В README таблица результатов начинается с Random Forest (0.7996).
Если запустить `notebooks/ModelTraining.ipynb` сверху вниз — числа совпадут с отчётом.
