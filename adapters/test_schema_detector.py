"""
adapters/test_schema_detector.py

Tests for the schema_detector and user_adapter normalization pipeline.

Run with:
    python -m pytest adapters/test_schema_detector.py -v
    -- or --
    python adapters/test_schema_detector.py

Each test group covers a distinct part of the pipeline:
    Group A: normalize_field_name()
    Group B: score_field() per target type
    Group C: detect_schema() on realistic messy column sets
    Group D: classify_fuel_label() for long-format CSV cell values
    Group E: end-to-end CSV ingestion via ingest()
"""

import io
import sys
import textwrap

import pandas as pd

try:
    import pytest
    _approx = _approx
except ImportError:
    pytest = None  # type: ignore[assignment]
    def _approx(x, **_):  # type: ignore[misc]
        return x

from adapters.schema_detector import (
    classify_fuel_label,
    detect_schema,
    normalize_field_name,
    score_field,
)
from adapters.user_adapter import ingest


# -
# Group A: normalize_field_name
# -

class TestNormalizeFieldName:

    def test_removes_unit_parentheses(self):
        assert normalize_field_name("Wind Generation (MW)") == "wind generation"

    def test_removes_unit_brackets(self):
        assert normalize_field_name("PV Power [kW]") == "pv power"

    def test_replaces_hyphens(self):
        assert normalize_field_name("WIND-GEN-HE") == "wind gen he"

    def test_replaces_underscores(self):
        assert normalize_field_name("solar_output_mw") == "solar output mw"

    def test_strips_timezone_parentheses(self):
        assert normalize_field_name("Date (MST)") == "date"

    def test_mixed_case(self):
        assert normalize_field_name("Hour Ending") == "hour ending"

    def test_settlement_timestamp(self):
        assert normalize_field_name("Settlement Timestamp") == "settlement timestamp"

    def test_already_clean(self):
        assert normalize_field_name("wind") == "wind"

    def test_numeric_string(self):
        # Should not crash on a number-only name
        result = normalize_field_name("12345")
        assert isinstance(result, str)

    def test_non_string_input(self):
        result = normalize_field_name(42)
        assert isinstance(result, str)

    def test_collapses_multiple_spaces(self):
        assert normalize_field_name("wind   gen   mw") == "wind gen mw"


# -
# Group B: score_field
# -

class TestScoreFieldWind:

    def _score(self, name):
        return score_field(name, "wind").score

    def test_exact_alias_wind(self):
        assert self._score("wind") == 1.0

    def test_wind_generation_mw(self):
        assert self._score("Wind Generation (MW)") == 1.0  # exact after normalize

    def test_wind_gen_he_hyphenated(self):
        assert self._score("WIND-GEN-HE") == 1.0

    def test_wind_output_underscore(self):
        assert self._score("wind_output") == 1.0

    def test_wind_power(self):
        assert self._score("Wind Power") == 1.0

    def test_messy_wind_generation(self):
        # Not in exact set — keyword + context path
        s = score_field("WIND GEN TOTAL", "wind")
        assert s.score >= 0.65
        assert any("wind" in r.lower() or "generation" in r.lower() or "total" in r.lower()
                   for r in s.reasons)

    def test_wind_forecast_penalized(self):
        # "forecast" should apply ambiguity penalty
        s = score_field("Wind Forecast (MW)", "wind")
        assert s.score < 0.50  # below accept threshold

    def test_wind_capacity_penalized(self):
        assert self._score("Wind Installed Capacity") < 0.50

    def test_wind_speed_suppressed(self):
        # Weather variable — should be suppressed below accept threshold
        assert self._score("Wind Speed") < 0.50

    def test_wind_direction_suppressed(self):
        assert self._score("wind_direction") < 0.50

    def test_wind_gust_suppressed(self):
        assert self._score("Wind Gust") < 0.50

    def test_completely_unrelated_field(self):
        assert self._score("Temperature (C)") == 0.0

    def test_solar_field_scores_zero_for_wind(self):
        assert self._score("Solar Generation (MW)") == 0.0

    def test_wind_actual_boosted(self):
        s = score_field("Wind Actual Generation", "wind")
        assert s.score >= 0.75


class TestScoreFieldSolar:

    def _score(self, name):
        return score_field(name, "solar").score

    def test_exact_alias_solar(self):
        assert self._score("solar") == 1.0

    def test_exact_alias_pv(self):
        assert self._score("PV") == 1.0

    def test_pv_power(self):
        assert self._score("PV Power") == 1.0

    def test_solar_output_underscore(self):
        assert self._score("solar_output") == 1.0

    def test_photovoltaic(self):
        assert self._score("Photovoltaic") == 1.0

    def test_solar_generation_mw(self):
        assert self._score("Solar Generation (MW)") == 1.0

    def test_pv_output_brackets(self):
        assert self._score("PV Output [MW]") == 1.0

    def test_solar_forecast_penalized(self):
        assert self._score("Solar Forecast") < 0.50

    def test_solar_irradiance_suppressed(self):
        assert self._score("Solar Irradiance") < 0.50

    def test_solar_radiation_suppressed(self):
        assert self._score("shortwave_radiation") == 0.0

    def test_wind_field_scores_zero_for_solar(self):
        assert self._score("Wind Generation (MW)") == 0.0


class TestScoreFieldTimestamp:

    def _score(self, name):
        return score_field(name, "timestamp").score

    def test_exact_timestamp(self):
        assert self._score("timestamp") == 1.0

    def test_exact_datetime(self):
        assert self._score("datetime") == 1.0

    def test_hour_ending(self):
        assert self._score("Hour Ending") == 1.0

    def test_settlement_timestamp(self):
        assert self._score("Settlement Timestamp") == 1.0

    def test_date_mst(self):
        # "(MST)" stripped -> "date" -> exact match
        assert self._score("Date (MST)") == 1.0

    def test_interval_start(self):
        assert self._score("Interval Start") == 1.0

    def test_date_time_utc(self):
        assert self._score("Date Time UTC") >= 0.70

    def test_he_abbreviation(self):
        assert self._score("HE") == 1.0  # "he" is in _TIMESTAMP_EXACT

    def test_settlement_date_utc(self):
        s = score_field("Settlement Date (UTC)", "timestamp")
        assert s.score >= 0.70

    def test_generation_value_not_timestamp(self):
        assert self._score("Wind Generation (MW)") == 0.0

    def test_numeric_column_not_timestamp(self):
        assert self._score("12345") == 0.0


# -
# Group C: detect_schema
# -

class TestDetectSchema:

    def test_clean_wide_columns(self):
        result = detect_schema(["timestamp", "Wind", "Solar", "temperature"])
        assert result.timestamp_col == "timestamp"
        assert result.wind_col      == "Wind"
        assert result.solar_col     == "Solar"
        assert result.needs_review  is False

    def test_messy_wide_columns(self):
        cols   = ["Hour Ending", "Wind Generation (MW)", "Solar_Output", "Region"]
        result = detect_schema(cols)
        assert result.timestamp_col == "Hour Ending"
        assert result.wind_col      == "Wind Generation (MW)"
        assert result.solar_col     == "Solar_Output"

    def test_hyphenated_wind_column(self):
        cols   = ["Date (MST)", "WIND-GEN-HE", "PV Output"]
        result = detect_schema(cols)
        assert result.timestamp_col == "Date (MST)"
        assert result.wind_col      == "WIND-GEN-HE"
        assert result.solar_col     == "PV Output"

    def test_wind_only_dataset(self):
        cols   = ["Settlement Timestamp", "Wind Power (MW)", "Region Code"]
        result = detect_schema(cols)
        assert result.timestamp_col is not None
        assert result.wind_col      is not None
        assert result.solar_col     is None

    def test_solar_only_dataset(self):
        cols   = ["datetime", "PV Power", "Location"]
        result = detect_schema(cols)
        assert result.wind_col  is None
        assert result.solar_col is not None

    def test_weather_columns_ignored(self):
        # Wind Speed and Solar Irradiance should NOT be detected as generation
        cols   = [
            "timestamp",
            "wind_speed_10m",       # weather — should be suppressed
            "shortwave_radiation",  # weather — no solar keyword
            "Wind Generation (MW)", # real generation
        ]
        result = detect_schema(cols)
        assert result.wind_col == "Wind Generation (MW)"
        assert result.solar_col is None

    def test_forecast_columns_ignored(self):
        cols = [
            "datetime",
            "Wind Forecast (MW)",
            "Wind Actual (MW)",
        ]
        result = detect_schema(cols)
        # "Wind Actual (MW)" should score higher than the penalized forecast
        assert result.wind_col == "Wind Actual (MW)"

    def test_no_usable_columns(self):
        cols   = ["Region", "Province", "Generator_ID"]
        result = detect_schema(cols)
        assert result.timestamp_col is None
        assert result.wind_col      is None
        assert result.solar_col     is None

    def test_all_scores_populated(self):
        cols   = ["timestamp", "Wind", "Solar"]
        result = detect_schema(cols)
        # 3 columns × 3 targets = 9 scores
        assert len(result.all_scores) == 9

    def test_column_not_reused_for_two_targets(self):
        # "wind" scores high for wind; make sure it's not also assigned as solar
        cols   = ["timestamp", "wind", "solar"]
        result = detect_schema(cols)
        assert result.wind_col != result.solar_col

    def test_settlement_hour_dataset(self):
        cols   = [
            "Settlement Hour",
            "Wind_Output_MW",
            "Solar-Power",
            "Demand",
        ]
        result = detect_schema(cols)
        assert result.timestamp_col is not None
        assert result.wind_col      is not None
        assert result.solar_col     is not None


# -
# Group D: classify_fuel_label
# -

class TestClassifyFuelLabel:

    def test_wind_exact(self):
        assert classify_fuel_label("WIND") == "WIND"

    def test_wind_lowercase(self):
        assert classify_fuel_label("wind") == "WIND"

    def test_wind_power(self):
        assert classify_fuel_label("Wind Power") == "WIND"

    def test_wind_turbine(self):
        assert classify_fuel_label("Wind Turbine") == "WIND"

    def test_wind_energy(self):
        assert classify_fuel_label("WIND ENERGY") == "WIND"

    def test_solar_exact(self):
        assert classify_fuel_label("SOLAR") == "SOLAR"

    def test_solar_lowercase(self):
        assert classify_fuel_label("solar") == "SOLAR"

    def test_pv(self):
        assert classify_fuel_label("PV") == "SOLAR"

    def test_photovoltaic(self):
        assert classify_fuel_label("photovoltaic") == "SOLAR"

    def test_solar_generation(self):
        assert classify_fuel_label("Solar Generation") == "SOLAR"

    def test_natural_gas_none(self):
        assert classify_fuel_label("Natural Gas") is None

    def test_nuclear_none(self):
        assert classify_fuel_label("Nuclear") is None

    def test_hydro_none(self):
        assert classify_fuel_label("Hydro") is None

    def test_empty_string_none(self):
        assert classify_fuel_label("") is None


# -
# Group E: end-to-end CSV ingestion
# -

def _csv(text: str) -> io.BytesIO:
    """Helper: convert a string of CSV content to a file-like BytesIO."""
    return io.BytesIO(textwrap.dedent(text).strip().encode())


class TestIngestCSV:

    def test_clean_wide_csv(self):
        buf = _csv("""
            timestamp,Wind,Solar
            2024-01-01 00:00,500,0
            2024-01-01 01:00,510,0
            2024-01-01 12:00,480,150
        """)
        df = ingest(buf, "csv")
        assert list(df.columns) == ["timestamp", "Wind", "Solar"]
        assert len(df) == 3
        assert df["Wind"].iloc[0] == _approx(500.0)

    def test_messy_wide_headers(self):
        buf = _csv("""
            Hour Ending,Wind Generation (MW),Solar_Output,Region
            2024-01-01 01:00,500,0,ON
            2024-01-01 02:00,510,0,ON
        """)
        df = ingest(buf, "csv")
        assert "Wind"  in df.columns
        assert "Solar" in df.columns
        assert len(df) == 2

    def test_hyphenated_and_mixed_case_headers(self):
        buf = _csv("""
            Date (MST),WIND-GEN-HE,PV Output
            2024-06-01 01:00,400,20
            2024-06-01 02:00,420,25
        """)
        df = ingest(buf, "csv")
        assert df["Wind"].notna().all()
        assert df["Solar"].notna().all()

    def test_pv_power_column_name(self):
        buf = _csv("""
            Settlement Timestamp,PV Power,Wind Power
            2024-07-01 10:00,200,300
            2024-07-01 11:00,210,305
        """)
        df = ingest(buf, "csv")
        assert df["Solar"].notna().all()
        assert df["Wind"].notna().all()

    def test_wind_only_csv(self):
        buf = _csv("""
            datetime,wind_output
            2024-01-01 00:00,700
            2024-01-01 01:00,720
        """)
        df = ingest(buf, "csv")
        assert df["Wind"].notna().all()
        assert df["Solar"].isna().all()

    def test_solar_only_csv(self):
        buf = _csv("""
            timestamp,solar_mw
            2024-06-01 09:00,50
            2024-06-01 10:00,80
        """)
        df = ingest(buf, "csv")
        assert df["Solar"].notna().all()
        assert df["Wind"].isna().all()

    def test_long_format_csv_aeso_style(self):
        buf = _csv("""
            Date (MST),Fuel Type,Volume
            2024-01-01 01:00,WIND,1200
            2024-01-01 01:00,SOLAR,0
            2024-01-01 02:00,WIND,1250
            2024-01-01 02:00,SOLAR,0
        """)
        df = ingest(buf, "csv")
        assert len(df) == 2  # pivoted: 2 unique hours
        assert df["Wind"].iloc[0] == _approx(1200.0)

    def test_long_format_fuzzy_fuel_labels(self):
        # Fuel labels use non-standard values — classify_fuel_label handles them
        buf = _csv("""
            Date,Fuel Type,Output
            2024-01-01 01:00,Wind Power,800
            2024-01-01 01:00,photovoltaic,30
        """)
        df = ingest(buf, "csv")
        assert df["Wind"].iloc[0]  == _approx(800.0)
        assert df["Solar"].iloc[0] == _approx(30.0)

    def test_deduplication_same_hour(self):
        buf = _csv("""
            timestamp,Wind,Solar
            2024-01-01 00:00,500,100
            2024-01-01 00:30,600,120
            2024-01-01 01:00,700,150
        """)
        df = ingest(buf, "csv")
        # 00:00 and 00:30 both floor to 00:00 -> averaged
        assert len(df) == 2
        assert df["Wind"].iloc[0] == _approx(550.0)

    def test_output_schema(self):
        buf = _csv("""
            timestamp,Wind,Solar
            2024-01-01 00:00,500,100
        """)
        df = ingest(buf, "csv")
        assert list(df.columns) == ["timestamp", "Wind", "Solar"]
        assert str(df["timestamp"].dtype) == "datetime64[ns]"

    def test_output_sorted_ascending(self):
        buf = _csv("""
            timestamp,Wind,Solar
            2024-01-01 03:00,300,0
            2024-01-01 01:00,100,0
            2024-01-01 02:00,200,0
        """)
        df = ingest(buf, "csv")
        ts = df["timestamp"].tolist()
        assert ts == sorted(ts)

    def test_no_duplicate_timestamps(self):
        buf = _csv("""
            timestamp,Wind,Solar
            2024-01-01 01:00,500,100
            2024-01-01 01:00,510,110
        """)
        df = ingest(buf, "csv")
        assert df["timestamp"].duplicated().sum() == 0


# -
# Standalone runner (no pytest required)
# -

if __name__ == "__main__":
    # Run a subset of representative cases and print a summary.
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    cases: list[tuple[str, bool]] = []

    def check(label: str, condition: bool) -> None:
        marker = PASS if condition else FAIL
        print(f"  {marker}  {label}")
        cases.append((label, condition))

    print("\n-- normalize_field_name --------------------------------------")
    check("Wind Generation (MW) -> 'wind generation'",
          normalize_field_name("Wind Generation (MW)") == "wind generation")
    check("WIND-GEN-HE -> 'wind gen he'",
          normalize_field_name("WIND-GEN-HE") == "wind gen he")
    check("solar_output_mw -> 'solar output mw'",
          normalize_field_name("solar_output_mw") == "solar output mw")
    check("Date (MST) -> 'date'",
          normalize_field_name("Date (MST)") == "date")
    check("PV Power [kW] -> 'pv power'",
          normalize_field_name("PV Power [kW]") == "pv power")

    print("\n- score_field: wind -")
    check("'Wind Generation (MW)' scores 1.0 for wind",
          score_field("Wind Generation (MW)", "wind").score == 1.0)
    check("'WIND-GEN-HE' scores 1.0 for wind",
          score_field("WIND-GEN-HE", "wind").score == 1.0)
    check("'Wind Speed' suppressed < 0.50 for wind",
          score_field("Wind Speed", "wind").score < 0.50)
    check("'Wind Forecast' penalized < 0.50 for wind",
          score_field("Wind Forecast (MW)", "wind").score < 0.50)

    print("\n- score_field: solar -")
    check("'PV Power' scores 1.0 for solar",
          score_field("PV Power", "solar").score == 1.0)
    check("'solar_output' scores 1.0 for solar",
          score_field("solar_output", "solar").score == 1.0)
    check("'Photovoltaic' scores 1.0 for solar",
          score_field("Photovoltaic", "solar").score == 1.0)
    check("'Solar Irradiance' suppressed < 0.50 for solar",
          score_field("Solar Irradiance", "solar").score < 0.50)

    print("\n- score_field: timestamp -")
    check("'Hour Ending' scores 1.0 for timestamp",
          score_field("Hour Ending", "timestamp").score == 1.0)
    check("'Settlement Timestamp' scores 1.0 for timestamp",
          score_field("Settlement Timestamp", "timestamp").score == 1.0)
    check("'Date (MST)' scores 1.0 for timestamp",
          score_field("Date (MST)", "timestamp").score == 1.0)

    print("\n- detect_schema -")
    r = detect_schema(["Hour Ending", "Wind Generation (MW)", "Solar_Output", "Extra"])
    check("Hour Ending -> timestamp_col",   r.timestamp_col == "Hour Ending")
    check("Wind Generation (MW) -> wind_col", r.wind_col == "Wind Generation (MW)")
    check("Solar_Output -> solar_col",      r.solar_col == "Solar_Output")

    r2 = detect_schema(["Date (MST)", "WIND-GEN-HE", "PV Output"])
    check("Date (MST) -> timestamp_col",  r2.timestamp_col == "Date (MST)")
    check("WIND-GEN-HE -> wind_col",      r2.wind_col == "WIND-GEN-HE")
    check("PV Output -> solar_col",       r2.solar_col == "PV Output")

    print("\n- classify_fuel_label -")
    check("'WIND' -> 'WIND'",         classify_fuel_label("WIND") == "WIND")
    check("'Wind Power' -> 'WIND'",   classify_fuel_label("Wind Power") == "WIND")
    check("'PV' -> 'SOLAR'",          classify_fuel_label("PV") == "SOLAR")
    check("'photovoltaic' -> 'SOLAR'",classify_fuel_label("photovoltaic") == "SOLAR")
    check("'Natural Gas' -> None",    classify_fuel_label("Natural Gas") is None)

    print("\n- end-to-end CSV ingest -")
    buf = _csv("""
        Hour Ending,Wind Generation (MW),Solar_Output
        2024-01-01 01:00,500,0
        2024-01-01 02:00,510,10
    """)
    df = ingest(buf, "csv")
    check("Wide CSV with messy headers: 2 rows returned",  len(df) == 2)
    check("Wide CSV: Wind column has values",              df["Wind"].notna().all())

    buf2 = _csv("""
        Date (MST),Fuel Type,Volume
        2024-01-01 01:00,WIND,1200
        2024-01-01 01:00,SOLAR,0
        2024-01-01 02:00,WIND,1250
        2024-01-01 02:00,SOLAR,5
    """)
    df2 = ingest(buf2, "csv")
    check("Long CSV: 2 pivoted rows returned", len(df2) == 2)
    check("Long CSV: Wind column has values",  df2["Wind"].notna().all())

    # Summary
    passed = sum(1 for _, ok in cases if ok)
    total  = len(cases)
    print(f"\n{'-' * 55}")
    print(f"  {passed}/{total} checks passed\n")
    if passed < total:
        sys.exit(1)
