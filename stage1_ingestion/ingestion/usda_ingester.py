"""
USDA NASS Quick Stats connector.
Pulls US-specific area, production, and farm-gate price data.
Requires free API key: https://quickstats.nass.usda.gov/api

Set env var: USDA_API_KEY=your_key
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from stage1_ingestion.config.settings import config
from stage1_ingestion.ingestion.base import BaseIngester


class USDAIngester(BaseIngester):
    SOURCE_NAME = "usda_nass"

    # Map USDA stat categories → internal metric names
    STAT_MAP = {
        "AREA HARVESTED": ("area_ha",        "HA",    0.404686),  # acres → HA
        "PRODUCTION":     ("production_mt",  "MT",    0.000907),  # short tons → MT
        "PRICE RECEIVED": ("price_usd_mt",   "USD/MT", 1.0),      # already USD
    }

    def __init__(self):
        super().__init__()
        self.cfg = config.usda

    def fetch_raw(self, **kwargs) -> List[Dict]:
        if self.cfg.api_key == "DEMO_KEY":
            self.log.warning(
                "Using DEMO_KEY — USDA results will be limited. "
                "Set USDA_API_KEY env var for full access."
            )

        records = []
        for commodity in self.cfg.commodities:
            for stat in self.cfg.stat_categories:
                self.log.info("Fetching USDA: %s / %s", commodity, stat)
                params = {
                    "key":              self.cfg.api_key,
                    "commodity_desc":   commodity,
                    "statisticcat_desc": stat,
                    "agg_level_desc":   "NATIONAL",
                    "freq_desc":        "ANNUAL",
                    "format":           "JSON",
                }
                try:
                    resp = self.get(
                        f"{self.cfg.base_url}/api_GET",
                        params=params,
                        timeout=self.cfg.timeout,
                    )
                    data = resp.json().get("data", [])
                    self.log.info("  → %d records", len(data))
                    records.extend(data)
                except Exception as exc:
                    self.log.error("USDA fetch failed (%s/%s): %s", commodity, stat, exc)

        return records

    def parse(self, raw: List[Dict]) -> pd.DataFrame:
        if not raw:
            self.log.warning("USDA returned empty payload")
            return pd.DataFrame()

        rows = []
        for rec in raw:
            stat_cat = rec.get("statisticcat_desc", "").upper()
            if stat_cat not in self.STAT_MAP:
                continue

            metric, unit, conversion = self.STAT_MAP[stat_cat]
            value_str = rec.get("Value", "").replace(",", "").strip()
            try:
                value = float(value_str) * conversion
            except (ValueError, TypeError):
                continue

            rows.append({
                "source":       self.SOURCE_NAME,
                "country":      "United States of America",
                "crop":         rec.get("commodity_desc", "").title(),
                "year":         int(rec.get("year", 0)),
                "metric":       metric,
                "value":        round(value, 4),
                "unit":         unit,
                "state":        rec.get("state_name", "NATIONAL"),
                "ingested_at":  datetime.utcnow(),
            })

        df = pd.DataFrame(rows)
        if df.empty:
             return df
        df = df[df["year"] >= 2000]
        df = df.drop_duplicates(subset=["country", "crop", "year", "metric", "state"])
        self.log.info("Parsed %d clean USDA rows", len(df))
        return df
