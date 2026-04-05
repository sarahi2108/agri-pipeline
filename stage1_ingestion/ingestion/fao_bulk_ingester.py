"""
FAO Bulk CSV ingester.
Reads the FAO bulk download CSV and filters to our crops and countries.
Use this when the FAO API is unavailable.

Download from:
https://fenixservices.fao.org/faostat/static/bulkdownloads/Production_Crops_Livestock_E_All_Data_(Normalized).zip
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stage1_ingestion.config.settings import config
from stage1_ingestion.ingestion.base import BaseIngester


class FAOBulkIngester(BaseIngester):
    SOURCE_NAME = "faostat"

    ELEMENT_MAP = {
        "Area harvested": ("area_ha",      "HA",     0.0001),   # ha → HA (same, just label)
        "Production":     ("production_mt", "MT",    0.001),    # tonnes → MT
        "Yield":          ("yield_mt_ha",  "MT/HA",  0.0001),   # hg/ha → MT/HA
    }

    # FAO uses hectares and tonnes — adjust conversion factors
    UNIT_CONVERSIONS = {
        "Area harvested": 1.0,        # already in HA
        "Production":     0.001,      # 1 tonne = 0.001 MT... actually 1 tonne = 1 MT
        "Yield":          1 / 10000,  # hg/ha → MT/HA
    }

    def __init__(self, csv_path: str = None):
        super().__init__()
        self.cfg = config.fao
        self.csv_path = Path(csv_path) if csv_path else None

    def fetch_raw(self, **kwargs) -> Any:
        """Read and filter the bulk CSV — no HTTP call needed."""
        csv_path = kwargs.get("csv_path") or self.csv_path
        if not csv_path or not Path(csv_path).exists():
            raise FileNotFoundError(
                f"FAO bulk CSV not found at: {csv_path}\n"
                "Download from: https://fenixservices.fao.org/faostat/static/bulkdownloads/"
                "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
            )

        self.log.info("Reading FAO bulk CSV: %s", csv_path)
        self.log.info("Filtering to %d crops and %d countries...",
                      len(self.cfg.crops), len(self.cfg.area_codes))

        # Read in chunks to avoid loading 4M rows into memory at once
        chunks = []
        chunk_size = 100_000
        total_read = 0

        for chunk in pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            encoding="latin-1",
            low_memory=False,
        ):
            total_read += len(chunk)
            filtered = chunk[
                chunk["Item"].isin(self.cfg.crops) &
                chunk["Area"].isin(self.cfg.area_codes) &
                chunk["Element"].isin(self.ELEMENT_MAP.keys()) &
                (chunk["Year"] >= self.cfg.start_year)
            ]
            if len(filtered) > 0:
                chunks.append(filtered)

        self.log.info("Scanned %d rows, kept %d matching rows",
                      total_read, sum(len(c) for c in chunks))

        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks, ignore_index=True)

    def parse(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            self.log.warning("FAO bulk CSV returned no matching rows")
            return pd.DataFrame()

        rows = []
        for _, rec in raw.iterrows():
            element = rec.get("Element", "")
            if element not in self.ELEMENT_MAP:
                continue

            metric, unit, conversion = self.ELEMENT_MAP[element]

            try:
                value = float(rec["Value"]) * conversion
            except (TypeError, ValueError):
                continue

            rows.append({
                "source":      self.SOURCE_NAME,
                "country":     rec["Area"],
                "crop":        rec["Item"],
                "year":        int(rec["Year"]),
                "metric":      metric,
                "value":       round(value, 4),
                "unit":        unit,
                "flag":        rec.get("Flag", ""),
                "ingested_at": datetime.utcnow(),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=["country", "crop", "year", "metric"])
        self.log.info("Parsed %d clean FAO rows from bulk CSV", len(df))
        return df

    def run(self, csv_path: str = None, **kwargs) -> pd.DataFrame:
        self.log.info("Starting FAO bulk ingestion")
        raw = self.fetch_raw(csv_path=csv_path or self.csv_path)
        df = self.parse(raw)
        if not df.empty:
            self.save(df, partition={"year": datetime.utcnow().year})
        return df
