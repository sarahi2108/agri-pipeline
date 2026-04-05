"""
Stage 2 — Data Transformation Layer
Uses DuckDB to clean, join, and harmonise raw Parquet data
from the Stage 1 data lake into a clean analytical table.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "stage2_transforms" / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("transform")


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def load_raw(con: duckdb.DuckDBPyConnection) -> None:
    fao_path = str(RAW_DIR / "faostat" / "year=2026" / "data.parquet")
    weather_path = str(RAW_DIR / "open_meteo_weather" / "year=2026" / "data.parquet")

    con.execute(f"""
        CREATE VIEW fao_raw AS 
        SELECT * EXCLUDE (year), year::INTEGER AS year
        FROM read_parquet('{fao_path}', hive_partitioning=false)
    """)
    con.execute(f"""
        CREATE VIEW weather_raw AS 
        SELECT * EXCLUDE (year), year::INTEGER AS year
        FROM read_parquet('{weather_path}', hive_partitioning=false)
    """)
    log.info("Registered FAO and weather Parquet files as DuckDB views")


def clean_fao(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE fao_clean AS
        SELECT
            country,
            crop,
            year,
            MAX(CASE WHEN metric = 'area_ha'        THEN value * 1000 END) AS area_ha,
            MAX(CASE WHEN metric = 'production_mt'  THEN value * 1000 END) AS production_mt,
            MAX(CASE WHEN metric = 'yield_mt_ha'    THEN value END) AS yield_mt_ha_reported
        FROM fao_raw
        WHERE value IS NOT NULL
          AND value > 0
          AND year >= 2000
        GROUP BY country, crop, year
    """)

    con.execute("""
        CREATE TABLE fao_final AS
        SELECT
            country,
            crop,
            year,
            area_ha,
            production_mt,
            yield_mt_ha_reported,
            CASE
                WHEN area_ha IS NOT NULL AND area_ha > 0 AND production_mt IS NOT NULL
                THEN ROUND(production_mt / area_ha, 4)
                ELSE yield_mt_ha_reported
            END AS yield_mt_ha
        FROM fao_clean
        WHERE area_ha IS NOT NULL OR production_mt IS NOT NULL
    """)

    n = con.execute("SELECT COUNT(*) FROM fao_final").fetchone()[0]
    log.info("FAO clean table: %d rows", n)


def clean_weather(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE weather_annual AS
        SELECT
            country,
            year,
            ROUND(AVG(CASE WHEN metric = 'temperature_2m_max' THEN value END), 2)
                AS avg_temp_max_c,
            ROUND(AVG(CASE WHEN metric = 'temperature_2m_min' THEN value END), 2)
                AS avg_temp_min_c,
            ROUND(SUM(CASE WHEN metric = 'precipitation_sum'  THEN value END), 1)
                AS total_precip_mm,
            ROUND(AVG(CASE WHEN metric = 'et0_fao_evapotranspiration' THEN value END), 2)
                AS avg_et0_mm,
            ROUND(AVG(CASE WHEN metric = 'temperature_2m_max'
                           AND in_growing_season = true THEN value END), 2)
                AS growing_season_temp_max_c,
            ROUND(SUM(CASE WHEN metric = 'precipitation_sum'
                           AND in_growing_season = true THEN value END), 1)
                AS growing_season_precip_mm
        FROM weather_raw
        WHERE year >= 2000
        GROUP BY country, year
    """)

    n = con.execute("SELECT COUNT(*) FROM weather_annual").fetchone()[0]
    log.info("Weather annual table: %d rows", n)


def join_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE analytical_base AS
        SELECT
            f.country,
            f.crop,
            f.year,
            f.area_ha,
            f.production_mt,
            f.yield_mt_ha,
            w.avg_temp_max_c,
            w.avg_temp_min_c,
            w.total_precip_mm,
            w.avg_et0_mm,
            w.growing_season_temp_max_c,
            w.growing_season_precip_mm,
            CURRENT_TIMESTAMP AS transformed_at
        FROM fao_final f
        LEFT JOIN weather_annual w
            ON f.country = w.country
            AND f.year   = w.year
        ORDER BY f.country, f.crop, f.year
    """)

    n = con.execute("SELECT COUNT(*) FROM analytical_base").fetchone()[0]
    log.info("Analytical base table: %d rows", n)


def save_output(con: duckdb.DuckDBPyConnection) -> Path:
    out_path = PROCESSED_DIR / "analytical_base.parquet"
    con.execute(f"""
        COPY analytical_base TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    log.info("Saved analytical base → %s", out_path)
    return out_path


def run_transforms() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Starting Stage 2 transformations")
    log.info("=" * 60)

    con = get_connection()
    load_raw(con)
    clean_fao(con)
    clean_weather(con)
    join_tables(con)
    out_path = save_output(con)

    df = pd.read_parquet(out_path)

    log.info("=" * 60)
    log.info("Stage 2 complete — %d rows in analytical base", len(df))
    log.info("Output: %s", out_path)
    log.info("=" * 60)

    con.close()
    return df


if __name__ == "__main__":
    df = run_transforms()
    print("\nSample output:")
    print(df[df["crop"] == "Grapes"].head(10).to_string())
    print("\nColumns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("\nYield range for Grapes (MT/HA):")
    grapes = df[df["crop"] == "Grapes"]["yield_mt_ha"]
    print(f"  Min: {grapes.min():.2f}  Max: {grapes.max():.2f}  Mean: {grapes.mean():.2f}")
