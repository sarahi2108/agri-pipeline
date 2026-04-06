"""
Stage 4 — FastAPI Serving Layer
Exposes yield forecasts, cluster assignments, and analytical data
via a REST API consumed by the Stage 5 dashboard.

Run:
    uvicorn stage4_api.main:app --reload --port 8000
"""

from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stage4_api.routers import yield_router, cluster_router

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "stage2_transforms" / "data" / "processed"


# ── App state — load data once at startup ─────────────────────────────────────
class AppState:
    features_df: pd.DataFrame = None
    clusters_df: pd.DataFrame = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load datasets into memory at startup."""
    state.features_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    state.clusters_df = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
    print(f"Loaded features: {state.features_df.shape}")
    print(f"Loaded clusters: {state.clusters_df.shape}")
    yield
    # Cleanup on shutdown
    state.features_df = None
    state.clusters_df = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agri Supply Chain Intelligence API",
    description="Yield forecasts, cluster assignments, and production analytics for global fresh produce.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(yield_router.router, prefix="/yield",   tags=["Yield"])
app.include_router(cluster_router.router, prefix="/clusters", tags=["Clusters"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "api": "Agri Supply Chain Intelligence",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "features_loaded": state.features_df is not None,
        "clusters_loaded": state.clusters_df is not None,
        "n_features_rows": len(state.features_df) if state.features_df is not None else 0,
    }


@app.get("/crops", tags=["Reference"])
def list_crops():
    """List all available crops."""
    crops = sorted(state.features_df["crop"].unique().tolist())
    return {"crops": crops}


@app.get("/countries", tags=["Reference"])
def list_countries():
    """List all available countries."""
    countries = sorted(state.features_df["country"].unique().tolist())
    return {"countries": countries}


@app.get("/summary", tags=["Analytics"])
def global_summary():
    """Global production summary — latest year available."""
    df = state.features_df
    latest = df[df["year"] == df["year"].max()]
    return {
        "latest_year": int(df["year"].max()),
        "total_production_mt": round(latest["production_mt"].sum(), 0),
        "avg_yield_mt_ha": round(latest["yield_mt_ha"].mean(), 2),
        "n_countries": int(latest["country"].nunique()),
        "n_crops": int(latest["crop"].nunique()),
    }
