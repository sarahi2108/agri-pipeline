"""
FAO STAT connector.
Pulls area harvested, production (MT), and yield data for
key fresh produce crops across all producing countries.

Docs: https://www.fao.org/faostat/en/#data
API:  https://fenixservices.fao.org/faostat/api/v1/en/
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from stage1_ingestion.config.settings import config
from stage1_ingestion.ingestion.base import BaseIngester


class FAOIngester(BaseIngester):
    SOURCE_NAME = "faostat"

    # Map FAO element names → our internal metric names + units
    ELEMENT_MAP = {
        "Area harvested":  ("area_ha",         "HA"),
        "Production":      ("production_mt",    "MT"),
        "Yield":           ("yield_hg_ha",      "hg/HA"),   # converted below
    }

    def __init__(self):
        super().__init__()
        self.cfg = config.fao

    def fetch_raw(self, **kwargs) -> List[Dict]:
        """
        FAO STAT bulk download endpoint returns JSON with a 'data' list.
        We request one call per element to stay within response size limits.
        """
        records = []
        for element in self.cfg.elements:
            self.log.info("Fetching FAO element: %s", element)
            params = {
                "area":         ",".join(self.cfg.area_codes),
                "item":         ",".join(self.cfg.crops),
                "element":      element,
                "year":         f"{self.cfg.start_year}:{datetime.utcnow().year}",
                "show_flags":   "false",
                "null_values":  "false",
                "output_type":  "objects",
            }
            try:
                resp = self.get(
                    f"{self.cfg.base_url}/en/data/QCL",
                    params=params,
                    timeout=self.cfg.timeout,
                )
                data = resp.json().get("data", [])
                self.log.info("  → %d records for '%s'", len(data), element)
                records.extend(data)
            except Exception as exc:
                self.log.error("FAO fetch failed for element '%s': %s", element, exc)

        return records

    def parse(self, raw: List[Dict]) -> pd.DataFrame:
        if not raw:
            self.log.warning("FAO returned empty payload")
            return pd.DataFrame()

        rows = []
        for rec in raw:
            element_label = rec.get("Element", "")
            if element_label not in self.ELEMENT_MAP:
                continue

            metric, unit = self.ELEMENT_MAP[element_label]
            value = self._safe_float(rec.get("Value"))
            if value is None:
                continue

            # Convert FAO yield from hg/HA → MT/HA (÷ 10,000)
            if metric == "yield_hg_ha":
                metric = "yield_mt_ha"
                unit = "MT/HA"
                value = round(value / 10_000, 4)

            rows.append({
                "source":       self.SOURCE_NAME,
                "country":      rec.get("Area", ""),
                "crop":         rec.get("Item", ""),
                "year":         int(rec.get("Year", 0)),
                "metric":       metric,
                "value":        value,
                "unit":         unit,
                "flag":         rec.get("Flag", ""),
                "ingested_at":  datetime.utcnow(),
            })

        df = pd.DataFrame(rows)
        if df.empty:
             return df
        df = df[df["year"] >= self.cfg.start_year]
        df = df.drop_duplicates(subset=["country", "crop", "year", "metric"])
        self.log.info("Parsed %d clean FAO rows", len(df))
        return df

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
