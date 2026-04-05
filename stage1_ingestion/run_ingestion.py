"""
run_ingestion.py
----------------
Run Stage 1 ingestion locally without Airflow — useful for development,
testing, and first-time data population.

Usage:
    python run_ingestion.py                  # all sources
    python run_ingestion.py --source fao     # single source
    python run_ingestion.py --source weather
    python run_ingestion.py --source usda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
import time
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from stage1_ingestion.ingestion.fao_ingester import FAOIngester
from stage1_ingestion.ingestion.weather_ingester import WeatherIngester
from stage1_ingestion.ingestion.usda_ingester import USDAIngester
from stage1_ingestion.ingestion.base import get_logger

log = get_logger("run_ingestion")


SOURCES = {
    "fao":     FAOIngester,
    "weather": WeatherIngester,
    "usda":    USDAIngester,
}


def run_source(name: str) -> int:
    log.info("=" * 60)
    log.info("Running source: %s", name.upper())
    log.info("=" * 60)
    t0 = time.perf_counter()
    ingester = SOURCES[name]()
    df = ingester.run()
    elapsed = time.perf_counter() - t0
    log.info("✓ %s complete — %d rows in %.1fs\n", name, len(df), elapsed)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 data ingestion")
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()) + ["all"],
        default="all",
        help="Which data source to ingest (default: all)",
    )
    args = parser.parse_args()

    targets = list(SOURCES.keys()) if args.source == "all" else [args.source]

    total_rows = 0
    errors = []

    for source in targets:
        try:
            total_rows += run_source(source)
        except Exception as exc:
            log.error("FAILED: %s — %s", source, exc)
            errors.append(source)

    log.info("=" * 60)
    log.info("Ingestion complete | Total rows: %d | Failed: %s",
             total_rows, errors if errors else "none")
    log.info("Raw data saved to: data/raw/")
    log.info("=" * 60)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
