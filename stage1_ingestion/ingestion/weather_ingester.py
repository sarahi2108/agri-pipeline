"""
Open-Meteo Historical Weather connector.
Pulls daily climate variables for key fresh produce production regions.
No API key required — free and open.

Docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from datetime import datetime, date
from typing import Any, Dict, List

import pandas as pd

from stage1_ingestion.config.settings import config
from stage1_ingestion.ingestion.base import BaseIngester


class WeatherIngester(BaseIngester):
    SOURCE_NAME = "open_meteo_weather"

    def __init__(self):
        super().__init__()
        self.cfg = config.weather

    def fetch_raw(self, **kwargs) -> Dict[str, Any]:
        """Fetch historical daily weather for all configured locations."""
        results = {}
        for location_name, (lat, lon) in self.cfg.locations.items():
            self.log.info("Fetching weather for %s (%.2f, %.2f)", location_name, lat, lon)
            params = {
                "latitude":         lat,
                "longitude":        lon,
                "start_date":       self.cfg.start_date,
                "end_date":         date.today().isoformat(),
                "daily":            ",".join(self.cfg.variables),
                "timezone":         "UTC",
            }
            try:
                resp = self.get(self.cfg.base_url, params=params, timeout=self.cfg.timeout)
                results[location_name] = resp.json()
                n_days = len(resp.json().get("daily", {}).get("time", []))
                self.log.info("  → %d days for %s", n_days, location_name)
            except Exception as exc:
                self.log.error("Weather fetch failed for %s: %s", location_name, exc)

        return results

    def parse(self, raw: Dict[str, Any]) -> pd.DataFrame:
        dfs = []
        for location_name, payload in raw.items():
            daily = payload.get("daily", {})
            if not daily or "time" not in daily:
                self.log.warning("No daily data for %s", location_name)
                continue

            df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
            for var in self.cfg.variables:
                if var in daily:
                    df[var] = daily[var]

            # Add growing season flag (Oct–Apr for Southern hemisphere; Apr–Oct Northern)
            lat = payload.get("latitude", 0)
            df["month"] = df["date"].dt.month
            if lat < 0:  # Southern hemisphere
                df["in_growing_season"] = df["month"].isin([10, 11, 12, 1, 2, 3, 4])
            else:
                df["in_growing_season"] = df["month"].isin([4, 5, 6, 7, 8, 9, 10])

            # Melt to long format matching standard schema
            id_vars = ["date", "month", "in_growing_season"]
            value_vars = [v for v in self.cfg.variables if v in df.columns]
            df_long = df.melt(id_vars=id_vars, value_vars=value_vars,
                              var_name="metric", value_name="value")
            df_long = df_long.dropna(subset=["value"])

            df_long["source"]   = self.SOURCE_NAME
            df_long["location"] = location_name
            df_long["lat"]      = payload.get("latitude")
            df_long["lon"]      = payload.get("longitude")
            df_long["year"]     = df_long["date"].dt.year
            df_long["country"]  = self._location_to_country(location_name)
            df_long["crop"]     = "all"  # weather is region-level, not crop-level
            df_long["unit"]     = df_long["metric"].map(self._unit_map())
            df_long["ingested_at"] = datetime.utcnow()

            dfs.append(df_long)

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)
        self.log.info("Parsed %d weather rows across %d locations", len(out), len(raw))
        return out

    @staticmethod
    def _location_to_country(location: str) -> str:
        mapping = {
            "Ica_Peru": "Peru",
            "Maule_Chile": "Chile",
            "Western_Cape_SA": "South Africa",
            "Murcia_Spain": "Spain",
            "California_US": "United States of America",
        }
        return mapping.get(location, location.split("_")[0])

    @staticmethod
    def _unit_map() -> Dict[str, str]:
        return {
            "temperature_2m_max":           "°C",
            "temperature_2m_min":           "°C",
            "precipitation_sum":            "mm",
            "et0_fao_evapotranspiration":   "mm",
            "soil_moisture_0_to_7cm":       "m³/m³",
        }
