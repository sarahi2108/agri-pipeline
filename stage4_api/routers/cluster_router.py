"""
Cluster router — serves regional cluster assignments and profiles.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/")
def get_clusters(
    crop: Optional[str] = Query(None, description="Filter by crop"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    """Return cluster assignments for all country/crop combinations."""
    from stage4_api.main import state
    df = state.clusters_df.copy()

    if crop:
        df = df[df["crop"] == crop]
    if country:
        df = df[df["country"] == country]
    if df.empty:
        raise HTTPException(status_code=404, detail="No clusters found for given filters")

    cols = ["country", "crop", "cluster", "cluster_label",
            "avg_yield", "yield_trend", "yield_volatility", "n_years"]
    available = [c for c in cols if c in df.columns]
    return {
        "records": df[available].to_dict(orient="records"),
        "n_records": len(df),
    }


@router.get("/summary")
def cluster_summary():
    """Summary profile for each cluster."""
    from stage4_api.main import state
    df = state.clusters_df.copy()

    summary = (
        df.groupby("cluster")
        .agg(
            n_regions=("country", "count"),
            avg_yield=("avg_yield", "mean"),
            yield_trend=("yield_trend", "mean"),
            yield_volatility=("yield_volatility", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return {"clusters": summary.to_dict(orient="records")}


@router.get("/country/{country}")
def country_clusters(country: str):
    """All cluster assignments for a specific country."""
    from stage4_api.main import state
    df = state.clusters_df[state.clusters_df["country"] == country]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for country: {country}")
    return {"country": country, "records": df.to_dict(orient="records")}
