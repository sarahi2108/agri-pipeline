"""
Yield router — serves historical yield data and model forecasts.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from xgboost import XGBRegressor

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "stage3_models" / "models"
PROCESSED_DIR = BASE_DIR / "stage2_transforms" / "data" / "processed"

_model_cache: dict = {}


def load_model(crop: str) -> XGBRegressor:
    """Load a trained yield model from disk, cached in memory."""
    key = crop.lower().replace(" ", "_").replace(",", "")
    if key not in _model_cache:
        path = MODELS_DIR / f"yield_{crop.lower().replace(' ', '_').replace(',', '')}.json"
        # Try alternate filename formats
        if not path.exists():
            slug = crop.lower().replace(" ", "_").replace(",", "")
            path = MODELS_DIR / f"yield_{slug}.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for crop: {crop}"
            )
        model = XGBRegressor()
        model.load_model(str(path))
        _model_cache[key] = model
    return _model_cache[key]


@router.get("/history")
def yield_history(
    crop: str = Query(..., description="Crop name e.g. Grapes"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    """Historical yield data for a crop, optionally filtered by country."""
    from stage4_api.main import state
    df = state.features_df.copy()
    df = df[df["crop"] == crop]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for crop: {crop}")
    if country:
        df = df[df["country"] == country]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {crop} in {country}")

    result = (
        df[["country", "crop", "year", "yield_mt_ha", "production_mt", "area_ha"]]
        .sort_values(["country", "year"])
        .dropna(subset=["yield_mt_ha"])
    )
    return {
        "crop": crop,
        "country": country or "all",
        "records": result.to_dict(orient="records"),
        "n_records": len(result),
    }


@router.get("/forecast")
def yield_forecast(
    crop: str = Query(..., description="Crop name e.g. Grapes"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    """
    Return model predictions vs actuals for the test period.
    Uses the trained XGBoost model for the specified crop.
    """
    from stage4_api.main import state
    from sklearn.preprocessing import LabelEncoder

    FEATURE_COLS = [
        "yield_lag_1", "yield_lag_2", "yield_lag_3",
        "yield_rolling_3y", "yield_rolling_5y", "yield_mt_ha_yoy_pct",
        "area_ha", "area_lag_1", "area_ha_yoy_pct",
        "avg_temp_max_c", "avg_temp_min_c", "total_precip_mm",
        "avg_et0_mm", "growing_season_temp_max_c", "growing_season_precip_mm",
        "avg_temp_max_c_anomaly", "total_precip_mm_anomaly",
        "growing_season_temp_max_c_anomaly", "growing_season_precip_mm_anomaly",
        "years_since_2000", "crop_year_rank", "country_encoded",
    ]

    df = state.features_df.copy()
    df = df[df["crop"] == crop].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for crop: {crop}")
    if country:
        df = df[df["country"] == country]

    le = LabelEncoder()
    df["country_encoded"] = le.fit_transform(df["country"])

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median())

    model = load_model(crop)
    df["predicted_yield"] = model.predict(X).round(2)
    df["residual"] = (df["yield_mt_ha"] - df["predicted_yield"]).round(2)

    result = df[["country", "crop", "year", "yield_mt_ha", "predicted_yield", "residual"]]
    result = result.sort_values(["country", "year"]).dropna(subset=["yield_mt_ha"])

    return {
        "crop": crop,
        "country": country or "all",
        "records": result.to_dict(orient="records"),
        "n_records": len(result),
    }


@router.get("/top")
def top_producers(
    crop: str = Query(..., description="Crop name"),
    metric: str = Query("production_mt", description="Metric: production_mt or yield_mt_ha"),
    top_n: int = Query(5, description="Number of top producers"),
):
    """Top N producing countries for a crop in the latest year."""
    from stage4_api.main import state
    df = state.features_df.copy()
    df = df[df["crop"] == crop]
    latest = df[df["year"] == df["year"].max()]
    top = latest.nlargest(top_n, metric)[["country", "crop", "year", metric]]
    return {
        "crop": crop,
        "metric": metric,
        "year": int(latest["year"].max()),
        "results": top.to_dict(orient="records"),
    }
