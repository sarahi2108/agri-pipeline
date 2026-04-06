"""
Stage 3 — Regional Clustering Model
Segments producing regions into clusters based on yield performance,
growth trajectory, and weather patterns.
Outputs cluster labels and a summary profile for each cluster.

Usage:
    python3 stage3_models/cluster.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import mlflow

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "stage2_transforms" / "data" / "processed" / "features.parquet"
PROCESSED_DIR = BASE_DIR / "stage2_transforms" / "data" / "processed"
PLOTS_DIR = BASE_DIR / "stage3_models" / "plots"
MLFLOW_DIR = BASE_DIR / "stage3_models" / "mlruns"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cluster")

# Cluster labels mapping — assigned after inspecting cluster profiles
CLUSTER_LABELS = {
    0: "High yield, stable",
    1: "High yield, volatile",
    2: "Emerging, growing",
    3: "Mature, flat",
}


def build_region_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the feature table to one row per country/crop combination.
    These profiles are what we cluster on.
    """
    profile = df.groupby(["country", "crop"]).agg(
        avg_yield          =("yield_mt_ha",         "mean"),
        yield_trend        =("yield_mt_ha_yoy_pct",  "mean"),
        yield_volatility   =("yield_mt_ha_yoy_pct",  "std"),
        avg_area           =("area_ha",              "mean"),
        area_growth        =("area_ha_yoy_pct",       "mean"),
        avg_production     =("production_mt",         "mean"),
        avg_temp           =("avg_temp_max_c",        "mean"),
        avg_precip         =("total_precip_mm",       "mean"),
        temp_anomaly_std   =("avg_temp_max_c_anomaly","std"),
        precip_anomaly_std =("total_precip_mm_anomaly","std"),
        n_years            =("year",                  "count"),
    ).reset_index()

    # Fill nulls with 0 for countries without weather data
    profile = profile.fillna(0)
    log.info("Built region profiles: %d country/crop combinations", len(profile))
    return profile


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """Use silhouette score to find optimal number of clusters."""
    scores = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(scores, key=scores.get)
    log.info("Silhouette scores: %s", {k: round(v, 3) for k, v in scores.items()})
    log.info("Optimal k: %d (score: %.3f)", best_k, scores[best_k])
    return best_k


def run_clustering(profile: pd.DataFrame, n_clusters: int = 4):
    """Fit KMeans and return cluster labels + scaler."""
    feature_cols = [
        "avg_yield", "yield_trend", "yield_volatility",
        "area_growth", "avg_temp", "avg_precip",
        "temp_anomaly_std", "precip_anomaly_std",
    ]

    X = profile[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k
    best_k = find_optimal_k(X_scaled, max_k=min(8, len(profile) - 1))

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    profile["cluster"] = km.fit_predict(X_scaled)
    profile["cluster_label"] = profile["cluster"].map(
        {i: f"Cluster {i}" for i in range(best_k)}
    )

    sil_score = silhouette_score(X_scaled, profile["cluster"])
    log.info("Final silhouette score: %.3f", sil_score)

    return profile, km, scaler, X_scaled, feature_cols, sil_score


def save_cluster_plot(profile: pd.DataFrame, X_scaled: np.ndarray) -> Path:
    """PCA scatter plot of clusters."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA scatter
    colors = ["#639922", "#D85A30", "#378ADD", "#888780", "#7F77DD", "#D4537E"]
    for cluster_id in sorted(profile["cluster"].unique()):
        mask = profile["cluster"] == cluster_id
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[cluster_id % len(colors)],
            label=f"Cluster {cluster_id}",
            s=80, alpha=0.8
        )
        for _, row in profile[mask].iterrows():
            idx = profile.index.get_loc(_)
            axes[0].annotate(
                f"{row['country'][:3]}-{row['crop'][:4]}",
                (coords[idx, 0], coords[idx, 1]),
                fontsize=6, alpha=0.7
            )

    axes[0].set_title("Regional clusters (PCA projection)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].legend()

    # Cluster profile heatmap
    cluster_means = profile.groupby("cluster")[
        ["avg_yield", "yield_trend", "yield_volatility", "area_growth", "avg_temp", "avg_precip"]
    ].mean()
    cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()

    im = axes[1].imshow(cluster_means_norm.T, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(len(cluster_means)))
    axes[1].set_xticklabels([f"Cluster {i}" for i in cluster_means.index])
    axes[1].set_yticks(range(len(cluster_means.columns)))
    axes[1].set_yticklabels(cluster_means.columns, fontsize=9)
    axes[1].set_title("Cluster profile heatmap (normalised)")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    out_path = PLOTS_DIR / "clusters.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Cluster plot saved → %s", out_path)
    return out_path


def print_cluster_summary(profile: pd.DataFrame) -> None:
    """Print a readable summary of each cluster."""
    print("\n" + "=" * 70)
    print("CLUSTER PROFILES")
    print("=" * 70)
    for cluster_id in sorted(profile["cluster"].unique()):
        members = profile[profile["cluster"] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(members)} regions)")
        print(f"  Avg yield:      {members['avg_yield'].mean():.1f} MT/HA")
        print(f"  Yield trend:    {members['yield_trend'].mean():+.1f}% YoY")
        print(f"  Volatility:     {members['yield_volatility'].mean():.1f}%")
        print(f"  Members:")
        for _, row in members.iterrows():
            print(f"    - {row['country']} / {row['crop']}")
    print("=" * 70)


def run_cluster_pipeline() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Starting regional clustering")
    log.info("=" * 60)

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("regional_clustering")

    df = pd.read_parquet(FEATURES_PATH)
    profile = build_region_profiles(df)

    with mlflow.start_run(run_name="kmeans_regional"):
        profile, km, scaler, X_scaled, feature_cols, sil_score = run_clustering(profile)

        mlflow.log_param("n_clusters", km.n_clusters)
        mlflow.log_param("feature_cols", feature_cols)
        mlflow.log_metric("silhouette_score", round(sil_score, 4))

        plot_path = save_cluster_plot(profile, X_scaled)
        mlflow.log_artifact(str(plot_path))

        # Save cluster assignments
        out_path = PROCESSED_DIR / "cluster_assignments.parquet"
        profile.to_parquet(out_path, index=False)
        mlflow.log_artifact(str(out_path))
        log.info("Cluster assignments saved → %s", out_path)

    print_cluster_summary(profile)

    log.info("=" * 60)
    log.info("Clustering complete — %d clusters, silhouette: %.3f",
             km.n_clusters, sil_score)
    log.info("=" * 60)

    return profile


if __name__ == "__main__":
    profile = run_cluster_pipeline()
