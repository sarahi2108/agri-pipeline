"""
Stage 3 — XGBoost Yield Forecaster
Trains a yield prediction model per crop using time-series cross validation.
Tracks all experiments with MLflow and generates SHAP explainability plots.

Usage:
    python3 stage3_models/train_yield.py                  # all crops
    python3 stage3_models/train_yield.py --crop Grapes    # single crop
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "stage2_transforms" / "data" / "processed" / "features.parquet"
MODELS_DIR = BASE_DIR / "stage3_models" / "models"
PLOTS_DIR = BASE_DIR / "stage3_models" / "plots"
MLFLOW_DIR = BASE_DIR / "stage3_models" / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_yield")

# ── Feature columns used by the model ────────────────────────────────────────
FEATURE_COLS = [
    "yield_lag_1",
    "yield_lag_2",
    "yield_lag_3",
    "yield_rolling_3y",
    "yield_rolling_5y",
    "yield_mt_ha_yoy_pct",
    "area_ha",
    "area_lag_1",
    "area_ha_yoy_pct",
    "avg_temp_max_c",
    "avg_temp_min_c",
    "total_precip_mm",
    "avg_et0_mm",
    "growing_season_temp_max_c",
    "growing_season_precip_mm",
    "avg_temp_max_c_anomaly",
    "total_precip_mm_anomaly",
    "growing_season_temp_max_c_anomaly",
    "growing_season_precip_mm_anomaly",
    "years_since_2000",
    "crop_year_rank",
    "country_encoded",
]

TARGET = "yield_mt_ha"


def load_data(crop: str) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["crop"] == crop].copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Encode country as integer
    le = LabelEncoder()
    df["country_encoded"] = le.fit_transform(df["country"])

    log.info("Loaded %d rows for crop: %s", len(df), crop)
    return df


def time_series_split(df: pd.DataFrame, test_years: int = 5):
    """
    Temporal train/test split — never use random splits on time-series data.
    Last N years = test set, everything before = train set.
    """
    max_year = df["year"].max()
    cutoff = max_year - test_years

    train = df[df["year"] <= cutoff].copy()
    test = df[df["year"] > cutoff].copy()

    log.info("Train: %d rows (%d–%d) | Test: %d rows (%d–%d)",
             len(train), train["year"].min(), train["year"].max(),
             len(test), test["year"].min(), test["year"].max())
    return train, test


def get_features_targets(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET].copy()
    # Fill nulls with median — XGBoost can handle NaN natively but
    # SHAP works better with imputed values
    X = X.fillna(X.median())
    return X, y


def tune_hyperparameters(X_train, y_train, n_trials: int = 30) -> dict:
    """Optuna hyperparameter search — finds best XGBoost params."""
    log.info("Tuning hyperparameters (%d trials)...", n_trials)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state":     42,
            "verbosity":        0,
        }
        model = XGBRegressor(**params)

        # Walk-forward cross validation
        years = sorted(X_train.index.map(lambda i: i))
        scores = []
        unique_years = sorted(set(years))

        if len(unique_years) < 6:
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            return mean_absolute_error(y_train, preds)

        for i in range(3, len(unique_years) - 1):
            tr_idx = [j for j, y in enumerate(years) if y <= unique_years[i]]
            val_idx = [j for j, y in enumerate(years) if y == unique_years[i + 1]]
            if not val_idx:
                continue
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))

        return np.mean(scores) if scores else float("inf")

    # Pass year index for walk-forward CV
    X_indexed = X_train.copy()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info("Best MAE from tuning: %.4f", study.best_value)
    return study.best_params


def train_model(X_train, y_train, params: dict) -> XGBRegressor:
    model = XGBRegressor(
        **params,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    metrics = {
        "mae":  round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "r2":   round(r2_score(y_test, preds), 4),
        "mape": round(np.mean(np.abs((y_test - preds) / y_test)) * 100, 2),
    }
    log.info("Test metrics — MAE: %.2f | RMSE: %.2f | R²: %.4f | MAPE: %.1f%%",
             metrics["mae"], metrics["rmse"], metrics["r2"], metrics["mape"])
    return metrics, preds


def save_shap_plot(model, X_test, crop: str) -> Path:
    """Generate and save SHAP summary plot."""
    log.info("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance — {crop} Yield Model")
    plt.tight_layout()

    out_path = PLOTS_DIR / f"shap_{crop.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP plot saved → %s", out_path)
    return out_path


def save_predictions_plot(y_test, preds, crop: str, test_df: pd.DataFrame) -> Path:
    """Plot actual vs predicted yield."""
    plt.figure(figsize=(12, 5))
    for country in test_df["country"].unique():
        mask = test_df["country"] == country
        plt.plot(test_df[mask]["year"], y_test[mask], "o-", label=f"{country} actual", alpha=0.7)
        plt.plot(test_df[mask]["year"], preds[mask], "s--", label=f"{country} predicted", alpha=0.7)

    plt.title(f"Actual vs Predicted Yield — {crop}")
    plt.xlabel("Year")
    plt.ylabel("Yield (MT/HA)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = PLOTS_DIR / f"predictions_{crop.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Predictions plot saved → %s", out_path)
    return out_path


def run_for_crop(crop: str, n_trials: int = 30) -> dict:
    """Full training pipeline for a single crop."""
    log.info("=" * 60)
    log.info("Training yield model for: %s", crop)
    log.info("=" * 60)

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("yield_forecaster")

    with mlflow.start_run(run_name=f"xgboost_{crop.lower().replace(' ', '_')}"):
        mlflow.set_tag("crop", crop)
        mlflow.set_tag("model_type", "XGBoost")

        df = load_data(crop)

        if len(df) < 20:
            log.warning("Not enough data for %s (%d rows) — skipping", crop, len(df))
            return {}

        train_df, test_df = time_series_split(df, test_years=4)
        X_train, y_train = get_features_targets(train_df)
        X_test, y_test = get_features_targets(test_df)

        mlflow.log_param("crop", crop)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("train_years", f"{train_df['year'].min()}-{train_df['year'].max()}")
        mlflow.log_param("test_years", f"{test_df['year'].min()}-{test_df['year'].max()}")

        best_params = tune_hyperparameters(X_train, y_train, n_trials=n_trials)
        mlflow.log_params(best_params)

        model = train_model(X_train, y_train, best_params)
        metrics, preds = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        shap_path = save_shap_plot(model, X_test, crop)
        pred_path = save_predictions_plot(y_test, preds, crop, test_df)
        mlflow.log_artifact(str(shap_path))
        mlflow.log_artifact(str(pred_path))

        model_path = MODELS_DIR / f"yield_{crop.lower().replace(' ', '_')}.json"
        model.save_model(str(model_path))
        log.info("Model saved → %s", model_path)

        return {"crop": crop, "metrics": metrics, "model_path": str(model_path)}


def main():
    parser = argparse.ArgumentParser(description="Train yield forecaster")
    parser.add_argument("--crop", type=str, default=None,
                        help="Crop to train on (default: all)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna hyperparameter tuning trials (default: 30)")
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_PATH)
    crops = [args.crop] if args.crop else df["crop"].unique().tolist()

    results = []
    for crop in crops:
        result = run_for_crop(crop, n_trials=args.trials)
        if result:
            results.append(result)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        m = r["metrics"]
        print(f"{r['crop']:<25} MAE: {m['mae']:>8.2f}  R²: {m['r2']:>6.4f}  MAPE: {m['mape']:>5.1f}%")
    print("=" * 60)
    print(f"\nMLflow UI: cd {BASE_DIR} && mlflow ui --backend-store-uri stage3_models/mlruns")


if __name__ == "__main__":
    main()
