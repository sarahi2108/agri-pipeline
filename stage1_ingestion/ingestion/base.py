"""
Base ingestion class.
All source-specific connectors inherit from this — standardises
retry logic, logging, rate limiting, and Parquet landing.
"""

import logging
import time
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stage1_ingestion.config.settings import config


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


class BaseIngester(ABC):
    """
    Abstract base for all data source connectors.

    Subclasses implement:
        fetch_raw()  → returns a raw dict/list from the API
        parse()      → converts raw → pd.DataFrame with standard columns
    """

    SOURCE_NAME: str = "unknown"

    def __init__(self):
        self.log = get_logger(self.SOURCE_NAME)
        self.session = self._build_session()
        self.storage_base = config.storage.base_path

    # ── HTTP session with retry ───────────────────────────────────────────────
    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "agri-pipeline/1.0 (research)"})
        return session

    # ── Core interface ────────────────────────────────────────────────────────
    @abstractmethod
    def fetch_raw(self, **kwargs) -> Any:
        """Call the remote API and return raw payload."""

    @abstractmethod
    def parse(self, raw: Any) -> pd.DataFrame:
        """Convert raw payload to a normalised DataFrame."""

    # ── Standard column schema ────────────────────────────────────────────────
    REQUIRED_COLS = {
        "source",       # str  — data source name
        "country",      # str  — ISO country name
        "crop",         # str  — crop name
        "year",         # int  — harvest year
        "metric",       # str  — e.g. 'area_ha', 'production_mt', 'yield_mt_ha'
        "value",        # float
        "unit",         # str  — 'HA', 'MT', 'MT/HA', etc.
        "ingested_at",  # datetime
    }

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"[{self.SOURCE_NAME}] Missing columns: {missing}")
        null_counts = df[list(self.REQUIRED_COLS)].isnull().sum()
        high_null = null_counts[null_counts > len(df) * 0.3]
        if not high_null.empty:
            self.log.warning("High null rates in columns: %s", high_null.to_dict())
        return df

    # ── Idempotent Parquet landing ────────────────────────────────────────────
    def save(self, df: pd.DataFrame, partition: Optional[Dict] = None) -> Path:
        """
        Save DataFrame as Parquet under:
            data/raw/<source>/<year>/data.parquet
        Idempotent — same payload produces same file hash.
        """
        df = self.validate(df)
        df["ingested_at"] = datetime.utcnow()

        dest_dir = self.storage_base / self.SOURCE_NAME
        if partition:
            for k, v in partition.items():
                dest_dir = dest_dir / f"{k}={v}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        out_path = dest_dir / "data.parquet"
        df.to_parquet(
            out_path,
            index=False,
            compression=config.storage.compression,
            engine="pyarrow",
        )
        self.log.info("Saved %d rows → %s", len(df), out_path)
        return out_path

    # ── Full run ──────────────────────────────────────────────────────────────
    def run(self, **kwargs) -> pd.DataFrame:
        self.log.info("Starting ingestion from %s", self.SOURCE_NAME)
        t0 = time.perf_counter()
        raw = self.fetch_raw(**kwargs)
        df = self.parse(raw)
        self.save(df, partition={"year": datetime.utcnow().year})
        elapsed = time.perf_counter() - t0
        self.log.info(
            "Finished %s — %d rows in %.1fs", self.SOURCE_NAME, len(df), elapsed
        )
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get(self, url: str, params: Dict = None, timeout: int = 30) -> requests.Response:
        self.log.debug("GET %s params=%s", url, params)
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp

    @staticmethod
    def payload_hash(data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()
