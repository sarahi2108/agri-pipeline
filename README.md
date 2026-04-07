# 🌍 Global Fresh Produce Supply Chain Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.10-0194E2?style=flat)](https://mlflow.org)

> An end-to-end ML engineering pipeline that ingests global crop production data, engineers predictive features from 26 years of agricultural and weather records, serves XGBoost yield forecasts via a REST API, and visualises insights through a live interactive dashboard.

**Live demo:** [https://agri-pipeline.sarahilyas.dev](https://agri-pipeline.sarahilyas.dev)  
**API docs:** [https://agri-pipeline.sarahilyas.dev/api/docs](https://agri-pipeline.sarahilyas.dev/api/docs)  
**GitHub:** [https://github.com/sarahi2108/agri-pipeline](https://github.com/sarahi2108/agri-pipeline)

---

## Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stage 1       │    │   Stage 2       │    │   Stage 3       │
│   Ingestion     │───▶│  Transforms     │───▶│   ML Models     │
│                 │    │                 │    │                 │
│ • FAO STAT API  │    │ • DuckDB SQL    │    │ • XGBoost x6    │
│ • USDA NASS     │    │ • Feature eng.  │    │ • K-Means       │
│ • Open-Meteo    │    │ • 41 features   │    │ • MLflow + SHAP │
│ • Airflow DAGs  │    │ • Parquet lake  │    │ • Optuna tuning │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐            │
│   Stage 5       │    │   Stage 4       │            │
│   Dashboard     │◀───│   REST API      │◀───────────┘
│                 │    │                 │
│ • Streamlit     │    │ • FastAPI       │
│ • Plotly charts │    │ • Docker-ready  │
│ • SHAP plots    │    │ • Swagger docs  │
│ • AWS + SSL     │    │ • CORS enabled  │
└─────────────────┘    └─────────────────┘
```

---

## Model Results

| Crop | R² | MAPE | MAE (MT/HA) |
|------|-----|------|-------------|
| Strawberries | **0.9909** | 4.5% | 11.78 |
| Tomatoes | 0.9668 | 3.2% | 21.32 |
| Citrus | 0.9661 | 3.8% | 7.80 |
| Grapes | 0.9320 | 9.8% | 11.37 |
| Avocados | 0.9374 | 7.8% | 5.89 |
| Blueberries | 0.8970 | 7.9% | 7.15 |

All models trained on 2000–2020 data, evaluated on 2021–2024 using **temporal cross-validation** — never random splits.

---

## Data Sources

| Source | Coverage | Records |
|--------|----------|---------|
| FAO STAT | 7 countries, 6 crops, 2000–2024 | 2,927 rows |
| Open-Meteo | 4 production regions, daily | 153,472 rows |
| USDA NASS | US domestic stats + prices | via API |

**Total data lake: 156,399 rows of real agricultural data**

---

## Tech Stack

**Data Engineering:** Python, Apache Airflow, DuckDB, Pandas, PyArrow, Parquet  
**Machine Learning:** XGBoost, Scikit-learn, SHAP, Optuna, MLflow  
**Serving:** FastAPI, Uvicorn, Pydantic  
**Visualisation:** Streamlit, Plotly, Folium  
**Infrastructure:** AWS EC2, Nginx, Let's Encrypt SSL, Ubuntu 22.04  

---

## Key Engineering Decisions

**Why temporal CV instead of random splits?**  
Random splits leak future yield data into training, artificially inflating metrics. In production you always predict forward in time — the evaluation must mirror that.

**Why DuckDB over Spark?**  
At under 200K rows, Spark adds infrastructure overhead with no performance benefit. DuckDB executes columnar SQL in-memory in under a second with zero setup.

**Why FAO bulk CSV over the API?**  
The FAO STAT API returned 521 errors during development — a real production outage. The bulk CSV fallback is a resilience pattern: same data, more reliable delivery.

**Why separate models per crop?**  
Yield dynamics differ significantly between crops — strawberries respond differently to temperature anomalies than grapes. A single model would underfit each crop's specific patterns.

---

## Project Structure

```
agri-pipeline/
├── stage1_ingestion/          # Data ingestion & Airflow DAGs
│   ├── ingestion/
│   │   ├── base.py            # Abstract base with retry + Parquet save
│   │   ├── fao_ingester.py    # FAO STAT API connector
│   │   ├── fao_bulk_ingester.py # Bulk CSV fallback
│   │   ├── weather_ingester.py  # Open-Meteo connector
│   │   └── usda_ingester.py   # USDA NASS connector
│   ├── dags/                  # Airflow DAG (monthly schedule)
│   └── tests/                 # 15 unit tests, all passing
├── stage2_transforms/         # DuckDB transformations
│   ├── transform.py           # Clean, join, pivot raw data
│   └── features.py            # 41 ML features engineered
├── stage3_models/             # ML training
│   ├── train_yield.py         # XGBoost + Optuna + MLflow + SHAP
│   └── cluster.py             # K-Means regional segmentation
├── stage4_api/                # FastAPI REST API
│   ├── main.py                # App + lifespan + CORS
│   └── routers/               # Yield + cluster endpoints
├── stage5_dashboard/          # Streamlit dashboard
│   └── app.py                 # 6 interactive charts + SHAP
├── data/
│   └── raw/                   # Partitioned Parquet data lake
└── requirements.txt
```

---

## Quickstart

```bash
# Clone and set up environment
git clone https://github.com/sarahi2108/agri-pipeline.git
cd agri-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run ingestion (FAO bulk CSV — download first)
python3 stage1_ingestion/run_ingestion.py --source weather
python3 -c "
from stage1_ingestion.ingestion.fao_bulk_ingester import FAOBulkIngester
FAOBulkIngester().run(csv_path='path/to/fao_bulk.csv')
"

# Run transformations
python3 stage2_transforms/transform.py
python3 stage2_transforms/features.py

# Train models
python3 stage3_models/train_yield.py --trials 20
python3 stage3_models/cluster.py

# Launch API
uvicorn stage4_api.main:app --reload --port 8000

# Launch dashboard
streamlit run stage5_dashboard/app.py

# View MLflow experiments
mlflow ui --backend-store-uri stage3_models/mlruns
```

---

## Tests

```bash
pytest stage1_ingestion/tests/ -v
# 15 passed in 1.3s
```




---

## About

Built by **Sarah Ilyas** — Data Scientist/ ML Engineer 

[LinkedIn](https://linkedin.com/in/sarahilyas) · [GitHub](https://github.com/sarahi2108)
