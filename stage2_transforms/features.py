"""
Stage 2 — Feature Engineering
Adds lag features, rolling averages, year-over-year changes,
and weather anomaly features to the analytical base table.
Output is the ML-ready feature table.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "stage2_transforms" / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("features")


def load_analytical_base() -> pd.DataFrame:
    path = PROCESSED_DIR / "analytical_base.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "analytical_base.parquet not found. Run transform.py first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded analytical base: %d rows", len(df))
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1, 2, and 3 year lags for yield, production, and area.
    Lags are computed within each country/crop group.
    These are the most important features for the yield forecaster.
    """
    df = df.sort_values(["country", "crop", "year"]).copy()

    for lag in [1, 2, 3]:
        df[f"yield_lag_{lag}"] = df.groupby(["country", "crop"])["yield_mt_ha"].shift(lag)
        df[f"production_lag_{lag}"] = df.groupby(["country", "crop"])["production_mt"].shift(lag)
        df[f"area_lag_{lag}"] = df.groupby(["country", "crop"])["area_ha"].shift(lag)

    log.info("Added lag features (1, 2, 3 years)")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 3-year and 5-year rolling averages for yield and weather.
    Rolling windows smooth out noise and capture medium-term trends.
    """
    df = df.sort_values(["country", "crop", "year"]).copy()

    for window in [3, 5]:
        df[f"yield_rolling_{window}y"] = (
            df.groupby(["country", "crop"])["yield_mt_ha"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            .round(4)
        )
        df[f"production_rolling_{window}y"] = (
            df.groupby(["country", "crop"])["production_mt"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            .round(2)
        )

    # Rolling weather averages (3-year)
    for col in ["avg_temp_max_c", "total_precip_mm", "growing_season_temp_max_c",
                "growing_season_precip_mm"]:
        if col in df.columns:
            df[f"{col}_rolling_3y"] = (
                df.groupby("country")[col]
                .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
                .round(2)
            )

    log.info("Added rolling average features (3y, 5y)")
    return df


def add_yoy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-over-year percentage changes for yield, production, area, and weather.
    Captures momentum and trend direction.
    """
    df = df.sort_values(["country", "crop", "year"]).copy()

    for col in ["yield_mt_ha", "production_mt", "area_ha"]:
        prev = df.groupby(["country", "crop"])[col].shift(1)
        df[f"{col}_yoy_pct"] = ((df[col] - prev) / prev * 100).round(2)

    for col in ["avg_temp_max_c", "total_precip_mm"]:
        if col in df.columns:
            prev = df.groupby("country")[col].shift(1)
            df[f"{col}_yoy_pct"] = ((df[col] - prev) / prev * 100).round(2)

    log.info("Added year-over-year change features")
    return df


def add_weather_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather anomalies: deviation from the country's long-run average.
    Anomalies are more predictive than raw values for yield models.
    """
    df = df.copy()

    for col in ["avg_temp_max_c", "total_precip_mm", "growing_season_temp_max_c",
                "growing_season_precip_mm"]:
        if col not in df.columns:
            continue
        country_mean = df.groupby("country")[col].transform("mean")
        country_std = df.groupby("country")[col].transform("std")
        df[f"{col}_anomaly"] = ((df[col] - country_mean) / country_std).round(3)

    log.info("Added weather anomaly features")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear trend features — position within the time series.
    Helps models distinguish early vs recent years.
    """
    df = df.copy()

    df["years_since_2000"] = df["year"] - 2000

    # Rank year within each country/crop (0 = earliest)
    df["crop_year_rank"] = df.groupby(["country", "crop"])["year"].rank(method="first") - 1

    log.info("Added trend features")
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in order."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_yoy_features(df)
    df = add_weather_anomaly_features(df)
    df = add_trend_features(df)
    return df


def save_features(df: pd.DataFrame) -> Path:
    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    log.info("Saved feature table → %s", out_path)
    return out_path


def run_feature_engineering() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Starting feature engineering")
    log.info("=" * 60)

    df = load_analytical_base()
    df = build_feature_table(df)
    out_path = save_features(df)

    log.info("=" * 60)
    log.info("Feature engineering complete")
    log.info("  Rows:     %d", len(df))
    log.info("  Columns:  %d", len(df.columns))
    log.info("  Output:   %s", out_path)
    log.info("=" * 60)

    return df


if __name__ == "__main__":
    df = run_feature_engineering()

    print("\nFeature table shape:", df.shape)
    print("\nAll columns:")
    for col in df.columns:
        nulls = df[col].isnull().sum()
        pct = round(nulls / len(df) * 100, 1)
        print(f"  {col:<45} nulls: {nulls} ({pct}%)")

    print("\nSample — Chile Grapes:")
    cols = ["country", "crop", "year", "yield_mt_ha", "yield_lag_1",
            "yield_rolling_3y", "yield_mt_ha_yoy_pct", "avg_temp_max_c_anomaly"]
    print(df[(df["country"] == "Chile") & (df["crop"] == "Grapes")][cols].to_string())
