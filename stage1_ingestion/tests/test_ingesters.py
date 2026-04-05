"""
Tests for Stage 1 ingestion.
Run: pytest tests/ -v

Tests use mocked HTTP responses so no real API calls are made.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── FAO tests ─────────────────────────────────────────────────────────────────
class TestFAOIngester:

    MOCK_FAO_RECORD = {
        "Area": "Peru",
        "Item": "Grapes",
        "Element": "Area harvested",
        "Year": "2022",
        "Value": "52000",
        "Flag": "A",
    }

    def test_parse_area_harvested(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        df = ingester.parse([self.MOCK_FAO_RECORD])
        assert len(df) == 1
        assert df.iloc[0]["metric"] == "area_ha"
        assert df.iloc[0]["unit"] == "HA"
        assert df.iloc[0]["value"] == 52000.0
        assert df.iloc[0]["country"] == "Peru"
        assert df.iloc[0]["crop"] == "Grapes"

    def test_parse_yield_converts_to_mt_ha(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        raw = [{**self.MOCK_FAO_RECORD, "Element": "Yield", "Value": "100000"}]
        df = ingester.parse(raw)
        assert df.iloc[0]["metric"] == "yield_mt_ha"
        assert df.iloc[0]["unit"] == "MT/HA"
        assert df.iloc[0]["value"] == pytest.approx(10.0, rel=1e-3)

    def test_parse_skips_null_values(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        raw = [{**self.MOCK_FAO_RECORD, "Value": None}]
        df = ingester.parse(raw)
        assert len(df) == 0

    def test_parse_skips_unknown_elements(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        raw = [{**self.MOCK_FAO_RECORD, "Element": "Unknown Element"}]
        df = ingester.parse(raw)
        assert len(df) == 0

    def test_deduplication(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        df = ingester.parse([self.MOCK_FAO_RECORD, self.MOCK_FAO_RECORD])
        assert len(df) == 1

    def test_validate_requires_standard_cols(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        bad_df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="Missing columns"):
            ingester.validate(bad_df)


# ── Weather tests ─────────────────────────────────────────────────────────────
class TestWeatherIngester:

    MOCK_WEATHER_PAYLOAD = {
        "Ica_Peru": {
            "latitude": -14.07,
            "longitude": -75.73,
            "daily": {
                "time": ["2022-01-01", "2022-01-02"],
                "temperature_2m_max": [28.5, 29.1],
                "temperature_2m_min": [15.2, 15.8],
                "precipitation_sum": [0.0, 2.1],
                "et0_fao_evapotranspiration": [4.5, 4.7],
                "soil_moisture_0_to_7cm": [0.22, 0.20],
            }
        }
    }

    def test_parse_produces_long_format(self):
        from stage1_ingestion.ingestion.weather_ingester import WeatherIngester
        ingester = WeatherIngester()
        df = ingester.parse(self.MOCK_WEATHER_PAYLOAD)
        # 2 days × 5 variables = 10 rows
        assert len(df) == 8
        assert "metric" in df.columns
        assert "value" in df.columns
        assert "location" in df.columns

    def test_country_mapping(self):
        from stage1_ingestion.ingestion.weather_ingester import WeatherIngester
        ingester = WeatherIngester()
        df = ingester.parse(self.MOCK_WEATHER_PAYLOAD)
        assert (df["country"] == "Peru").all()

    def test_growing_season_flag(self):
        from stage1_ingestion.ingestion.weather_ingester import WeatherIngester
        ingester = WeatherIngester()
        df = ingester.parse(self.MOCK_WEATHER_PAYLOAD)
        # January is in growing season for Southern hemisphere (lat < 0)
        assert df["in_growing_season"].all()

    def test_empty_payload(self):
        from stage1_ingestion.ingestion.weather_ingester import WeatherIngester
        ingester = WeatherIngester()
        df = ingester.parse({})
        assert len(df) == 0


# ── USDA tests ────────────────────────────────────────────────────────────────
class TestUSDAIngester:

    MOCK_USDA_RECORD = {
        "commodity_desc": "GRAPES",
        "statisticcat_desc": "AREA HARVESTED",
        "year": "2022",
        "Value": "850,000",   # acres
        "state_name": "NATIONAL",
    }

    def test_parse_converts_acres_to_ha(self):
        from stage1_ingestion.ingestion.usda_ingester import USDAIngester
        ingester = USDAIngester()
        df = ingester.parse([self.MOCK_USDA_RECORD])
        assert len(df) == 1
        assert df.iloc[0]["unit"] == "HA"
        expected_ha = 850_000 * 0.404686
        assert df.iloc[0]["value"] == pytest.approx(expected_ha, rel=1e-3)

    def test_parse_skips_bad_values(self):
        from stage1_ingestion.ingestion.usda_ingester import USDAIngester
        ingester = USDAIngester()
        bad = [{**self.MOCK_USDA_RECORD, "Value": "(D)"}]  # suppressed data
        df = ingester.parse(bad)
        assert len(df) == 0

    def test_country_always_usa(self):
        from stage1_ingestion.ingestion.usda_ingester import USDAIngester
        ingester = USDAIngester()
        df = ingester.parse([self.MOCK_USDA_RECORD])
        assert df.iloc[0]["country"] == "United States of America"


# ── Base ingester tests ───────────────────────────────────────────────────────
class TestBaseIngester:

    def test_session_has_retry(self):
        from stage1_ingestion.ingestion.fao_ingester import FAOIngester
        ingester = FAOIngester()
        adapter = ingester.session.get_adapter("https://")
        assert adapter.max_retries.total == 3

    def test_payload_hash_is_deterministic(self):
        from stage1_ingestion.ingestion.base import BaseIngester
        data = {"key": "value", "num": 42}
        h1 = BaseIngester.payload_hash(data)
        h2 = BaseIngester.payload_hash(data)
        assert h1 == h2
        assert len(h1) == 32
