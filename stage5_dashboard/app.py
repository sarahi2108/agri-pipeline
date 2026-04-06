"""
Stage 5 — Streamlit Dashboard
Global Fresh Produce Supply Chain Intelligence Platform
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "stage2_transforms" / "data" / "processed"
PLOTS_DIR = BASE_DIR / "stage3_models" / "plots"

st.set_page_config(
    page_title="Agri Supply Chain Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    clusters = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
    return features, clusters


features_df, clusters_df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🌱 Filters")

all_crops = sorted(features_df["crop"].unique().tolist())
selected_crop = st.sidebar.selectbox("Crop", all_crops, index=all_crops.index("Grapes"))

all_countries = sorted(features_df["country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries", all_countries, default=all_countries
)

year_min = int(features_df["year"].min())
year_max = int(features_df["year"].max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("---")
st.sidebar.markdown("**Data sources**")
st.sidebar.markdown("- FAO STAT (2000–2024)")
st.sidebar.markdown("- Open-Meteo weather")
st.sidebar.markdown("- XGBoost yield models")

# ── Filter data ───────────────────────────────────────────────────────────────
df = features_df[
    (features_df["crop"] == selected_crop) &
    (features_df["country"].isin(selected_countries)) &
    (features_df["year"].between(*year_range))
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌍 Global Fresh Produce Supply Chain Intelligence")
st.markdown(f"**{selected_crop}** · {year_range[0]}–{year_range[1]} · {len(selected_countries)} countries")
st.divider()

# ── Metric cards ─────────────────────────────────────────────────────────────
latest = df[df["year"] == df["year"].max()]
prev = df[df["year"] == df["year"].max() - 1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    val = latest["production_mt"].sum() / 1_000_000
    delta = (latest["production_mt"].sum() - prev["production_mt"].sum()) / prev["production_mt"].sum() * 100
    st.metric("Total production", f"{val:.1f}M MT", f"{delta:+.1f}%")

with col2:
    val = latest["yield_mt_ha"].mean()
    delta = latest["yield_mt_ha"].mean() - prev["yield_mt_ha"].mean()
    st.metric("Avg yield", f"{val:.1f} MT/HA", f"{delta:+.1f}")

with col3:
    val = latest["area_ha"].sum() / 1000
    st.metric("Total area", f"{val:.0f}K HA", "")

with col4:
    yoy = df[df["year"] == df["year"].max()]["yield_mt_ha_yoy_pct"].mean()
    st.metric("Yield YoY", f"{yoy:+.1f}%", "")

st.divider()

# ── Row 1: Yield over time + Production bar ───────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Yield over time (MT/HA)")
    fig = px.line(
        df.dropna(subset=["yield_mt_ha"]),
        x="year", y="yield_mt_ha", color="country",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=320, margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="", yaxis_title="MT/HA",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Production by country — latest year")
    fig2 = px.bar(
        latest.sort_values("production_mt", ascending=True),
        x="production_mt", y="country",
        orientation="h",
        color="production_mt",
        color_continuous_scale="Greens",
        labels={"production_mt": "MT", "country": ""},
    )
    fig2.update_layout(
        height=320, margin=dict(t=10, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Weather correlation + YoY change ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Yield vs growing season temperature")
    scatter_df = df.dropna(subset=["yield_mt_ha", "growing_season_temp_max_c"])
    if not scatter_df.empty:
        fig3 = px.scatter(
            scatter_df,
            x="growing_season_temp_max_c", y="yield_mt_ha",
            color="country", size="production_mt",
            hover_data=["year"],
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                "growing_season_temp_max_c": "Growing season temp (°C)",
                "yield_mt_ha": "Yield (MT/HA)",
            }
        )
        fig3.update_layout(height=300, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No weather data available for selected countries.")

with col2:
    st.subheader("Year-over-year yield change (%)")
    yoy_df = df.dropna(subset=["yield_mt_ha_yoy_pct"])
    if not yoy_df.empty:
        fig4 = px.box(
            yoy_df,
            x="country", y="yield_mt_ha_yoy_pct",
            color="country",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"yield_mt_ha_yoy_pct": "YoY %", "country": ""},
        )
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(height=300, margin=dict(t=10, b=10),
                           showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Row 3: Cluster map + SHAP plot ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regional clusters")
    crop_clusters = clusters_df[clusters_df["crop"] == selected_crop]
    if not crop_clusters.empty:
        fig5 = px.scatter(
            crop_clusters,
            x="avg_yield", y="yield_volatility",
            color="cluster_label", size="avg_production",
            hover_data=["country", "yield_trend", "n_years"],
            text="country",
            color_discrete_sequence=["#639922", "#D85A30", "#378ADD", "#888780"],
            labels={
                "avg_yield": "Avg yield (MT/HA)",
                "yield_volatility": "Yield volatility (%)",
                "cluster_label": "Cluster",
            },
        )
        fig5.update_traces(textposition="top center", textfont_size=9)
        fig5.update_layout(height=320, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No cluster data for selected crop.")

with col2:
    st.subheader("SHAP feature importance")
    crop_slug = selected_crop.lower().replace(" ", "_").replace(",", "")
    shap_path = PLOTS_DIR / f"shap_{crop_slug}.png"
    if shap_path.exists():
        st.image(str(shap_path), use_column_width=True)
    else:
        st.info(f"No SHAP plot found for {selected_crop}.")

st.divider()

# ── Row 4: Raw data table ─────────────────────────────────────────────────────
with st.expander("View raw data"):
    display_cols = ["country", "crop", "year", "area_ha", "production_mt",
                    "yield_mt_ha", "avg_temp_max_c", "total_precip_mm",
                    "yield_mt_ha_yoy_pct", "yield_rolling_3y"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].sort_values(["country", "year"], ascending=[True, False]),
        use_container_width=True,
        height=300,
    )

st.caption("Data: FAO STAT · Open-Meteo · USDA NASS | Models: XGBoost + K-Means | Built with Streamlit")
