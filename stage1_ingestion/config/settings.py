"""
Global configuration for the Agri Supply Chain Intelligence Pipeline.
All secrets are loaded from environment variables — never hardcoded.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_RAW, DATA_PROCESSED, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── FAO STAT ──────────────────────────────────────────────────────────────────
@dataclass
class FAOConfig:
    base_url: str = "https://fenixservices.fao.org/faostat/api/v1"
    crops: List[str] = field(default_factory=lambda: [
        "Grapes", "Blueberries", "Avocados", "Tomatoes",
        "Strawberries", "Citrus Fruit, Total"
    ])
    elements: List[str] = field(default_factory=lambda: [
        "Area harvested",   # HA
        "Production",       # MT
        "Yield",            # hg/ha — we'll convert to MT/HA
    ])
    # Key producing regions of interest
    area_codes: List[str] = field(default_factory=lambda: [
        "Peru", "Chile", "South Africa", "Spain",
        "Italy", "United States of America", "China"
    ])
    start_year: int = 2000
    timeout: int = 30


# ── USDA NASS ─────────────────────────────────────────────────────────────────
@dataclass
class USDAConfig:
    base_url: str = "https://quickstats.nass.usda.gov/api"
    api_key: str = field(default_factory=lambda: os.getenv("USDA_API_KEY", "DEMO_KEY"))
    commodities: List[str] = field(default_factory=lambda: ["GRAPES", "BLUEBERRIES", "STRAWBERRIES"])
    stat_categories: List[str] = field(default_factory=lambda: ["AREA HARVESTED", "PRODUCTION", "PRICE RECEIVED"])
    timeout: int = 30


# ── Open-Meteo weather ────────────────────────────────────────────────────────
@dataclass
class WeatherConfig:
    base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    # Key production region coordinates {name: (lat, lon)}
    locations: dict = field(default_factory=lambda: {
        "Ica_Peru":         (-14.07, -75.73),
        "Maule_Chile":      (-35.43, -71.67),
        "Western_Cape_SA":  (-33.93, 18.86),
        "Murcia_Spain":     (37.98, -1.13),
        "California_US":    (36.78, -119.42),
    })
    variables: List[str] = field(default_factory=lambda: [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "soil_moisture_0_to_7cm",
    ])
    start_date: str = "2000-01-01"
    timeout: int = 60


# ── World Bank commodity prices ───────────────────────────────────────────────
@dataclass
class WorldBankConfig:
    base_url: str = "https://api.worldbank.org/v2"
    # Commodity price indicators (grapes/produce proxies)
    indicators: List[str] = field(default_factory=lambda: [
        "PGNUTS",   # groundnuts — proxy for soft commodity prices
        "PFOOD",    # food price index
    ])
    timeout: int = 30


# ── Storage / data lake ───────────────────────────────────────────────────────
@dataclass
class StorageConfig:
    # Local Parquet data lake (swap base_path to S3 URI for cloud)
    base_path: Path = DATA_RAW
    partition_cols: List[str] = field(default_factory=lambda: ["source", "year"])
    compression: str = "snappy"

    # MinIO / S3 (optional — set USE_S3=true in env to activate)
    use_s3: bool = field(default_factory=lambda: os.getenv("USE_S3", "false").lower() == "true")
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "agri-pipeline-raw"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-west-1"))


# ── Airflow / scheduler ───────────────────────────────────────────────────────
@dataclass
class SchedulerConfig:
    # Cron: monthly refresh (data rarely updates more frequently)
    schedule_interval: str = "0 3 1 * *"
    retries: int = 3
    retry_delay_minutes: int = 10
    email_on_failure: bool = False
    catchup: bool = False


# ── Root config object ────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    fao: FAOConfig = field(default_factory=FAOConfig)
    usda: USDAConfig = field(default_factory=USDAConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    worldbank: WorldBankConfig = field(default_factory=WorldBankConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# Singleton
config = PipelineConfig()
