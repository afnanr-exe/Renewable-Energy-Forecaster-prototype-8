"""
adapters/user_adapter.py

User-data ingestion and normalization pipeline.

Accepts raw file objects (CSV or XML) and returns a canonical DataFrame
with the schema expected by all downstream pipelines:

    timestamp   datetime64[ns]  tz-naive, floored to the hour
    Wind        float64 or NaN
    Solar       float64 or NaN

Schema detection is powered by adapters.schema_detector, which uses a
scoring-based approach so messy, abbreviated, or non-standard column names
are handled automatically without requiring manual renaming by the user.

Supported input formats
-----------------------
CSV (wide)      Wind and/or Solar appear as separate columns.
                e.g. "timestamp, Wind Generation (MW), Solar_Output, ..."

CSV (long)      AESO-style: one row per (timestamp, fuel_type, value).
                e.g. "Date (MST), Fuel Type, Volume, ..."
                Fuel-type cell values are also fuzzy-matched
                (e.g. "Wind Power", "WIND TURBINE", "photovoltaic").

XML (IESO)      http://www.ieso.ca/schema — DailyData/HourlyData/FuelTotal.
                Parsed with a dedicated, namespace-aware handler.

XML (generic)   Any repeating-record XML. Records are flattened into a
                DataFrame and schema detection is applied to the resulting
                column names.

Public API
----------
ingest(file_obj, file_format)               → pd.DataFrame
ingest_with_metadata(file_obj, file_format) → IngestResult
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from adapters.schema_detector import (
    SchemaDetection,
    classify_fuel_label,
    detect_schema,
    normalize_field_name,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """
    Rich result returned by ingest_with_metadata().

    Attributes
    ----------
    df            : canonical DataFrame[timestamp, Wind, Solar]
    detection     : full SchemaDetection with per-column scores and reasons
    source_format : "wide_csv" | "long_csv" | "ieso_xml" | "generic_xml"
    warnings      : combined list of all warnings from detection and parsing
    needs_review  : True when any detection was ambiguous or low-confidence
    """
    df:            pd.DataFrame
    detection:     SchemaDetection
    source_format: str
    warnings:      list[str] = field(default_factory=list)
    needs_review:  bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp normalization (shared by all parsers)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse a Series of timestamp strings or objects to datetime64[ns], tz-naive,
    floored to the hour boundary.

    Raises ValueError with a descriptive message on parse failure.
    """
    try:
        parsed = pd.to_datetime(series)
    except Exception as exc:
        raise ValueError(f"Cannot parse timestamp column: {exc}") from exc

    # Strip timezone info so the result matches tz-naive weather data
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)

    return parsed.dt.floor("h")


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsers
# ─────────────────────────────────────────────────────────────────────────────

# Aliases used to locate the fuel-type column in long-format CSVs.
# Not handled by schema_detector (which targets generation columns).
_FUEL_TYPE_ALIASES: frozenset[str] = frozenset({
    "fuel type", "fuel_type", "fueltype", "fuel", "type",
    "energy type", "generation type", "resource type", "resource",
})

# Aliases for the numeric value column in long-format CSVs.
_VALUE_ALIASES: frozenset[str] = frozenset({
    "volume", "output", "mw", "mwh", "value", "generation",
    "energy", "power", "amount", "quantity", "actual",
    "generation mw", "output mw", "total mw",
})


def _parse_wide_csv(df_raw: pd.DataFrame, detection: SchemaDetection) -> pd.DataFrame:
    """
    Handle CSVs where Wind and/or Solar are already separate named columns.

    Uses the SchemaDetection result to identify and rename columns.
    Deduplicates same-hour entries with mean aggregation (safe for power readings).

    Raises ValueError if no timestamp column was detected.
    """
    if detection.timestamp_col is None:
        raise ValueError(
            f"Cannot find a timestamp column in wide-format CSV. "
            f"Columns found: {df_raw.columns.tolist()}. "
            f"Expected names like: timestamp, date, datetime, hour, "
            f"'Date (MST)', 'Interval Start', 'Settlement Datetime', etc."
        )

    df = df_raw.copy()
    df["timestamp"] = _normalise_timestamps(df[detection.timestamp_col])

    if detection.wind_col:
        df["Wind"] = pd.to_numeric(df[detection.wind_col], errors="coerce")
    if detection.solar_col:
        df["Solar"] = pd.to_numeric(df[detection.solar_col], errors="coerce")

    agg: dict[str, str] = {}
    if "Wind"  in df.columns: agg["Wind"]  = "mean"
    if "Solar" in df.columns: agg["Solar"] = "mean"

    return df.groupby("timestamp", as_index=False).agg(agg)


def _parse_long_csv(df_raw: pd.DataFrame, detection: SchemaDetection) -> pd.DataFrame:
    """
    Handle long-format CSVs (e.g. AESO-style) with one row per
    (timestamp, fuel_type, value). Pivots to wide format and uses sum to
    aggregate multiple generators reporting for the same hour.

    Fuel-type cell values are classified via classify_fuel_label() so that
    labels like "Wind Turbine", "WIND-POWER", "Photovoltaic", and "PV" are
    all matched correctly.

    Raises ValueError if required columns cannot be found or no usable rows remain.
    """
    cols = df_raw.columns.tolist()

    # Timestamp column from schema detection
    ts_col = detection.timestamp_col

    # Fuel-type column: exact-ish match on normalized alias
    fuel_col = next(
        (c for c in cols if normalize_field_name(c) in _FUEL_TYPE_ALIASES),
        None,
    )

    # Value column: exact-ish match on normalized alias
    value_col = next(
        (c for c in cols if normalize_field_name(c) in _VALUE_ALIASES),
        None,
    )

    if ts_col is None or fuel_col is None or value_col is None:
        raise ValueError(
            f"Long-format CSV requires a timestamp column, a fuel-type column, "
            f"and a numeric value column. "
            f"Detected — timestamp: '{ts_col}', fuel_type: '{fuel_col}', "
            f"value: '{value_col}'. "
            f"Columns in file: {cols}"
        )

    df = df_raw[[ts_col, fuel_col, value_col]].copy()
    df.columns = ["timestamp", "_fuel", "_value"]

    df["timestamp"] = _normalise_timestamps(df["timestamp"])
    df["_value"]    = pd.to_numeric(df["_value"], errors="coerce")

    # Fuzzy-classify each fuel label to WIND / SOLAR / None
    df["_category"] = df["_fuel"].astype(str).apply(classify_fuel_label)
    df = df[df["_category"].notna()].copy()

    if df.empty:
        distinct = list(df_raw[fuel_col].dropna().unique())[:20]
        raise ValueError(
            f"Long-format CSV: no rows with recognizable Wind or Solar fuel labels "
            f"in column '{fuel_col}'. "
            f"Distinct labels found: {distinct}. "
            f"Expected labels like: 'WIND', 'wind power', 'solar', 'PV', "
            f"'photovoltaic', 'Wind Turbine', etc."
        )

    pivoted = (
        df.groupby(["timestamp", "_category"])["_value"]
          .sum()
          .unstack("_category")
          .reset_index()
    )
    pivoted.columns.name = None

    rename = {}
    if "WIND"  in pivoted.columns: rename["WIND"]  = "Wind"
    if "SOLAR" in pivoted.columns: rename["SOLAR"] = "Solar"

    return pivoted.rename(columns=rename)


def _parse_csv_obj(file_obj) -> IngestResult:
    """
    Read a CSV file object, auto-detect its schema, and parse to canonical form.

    Detection order
    ---------------
    1. Wide  — detect_schema() finds a Wind or Solar column directly
    2. Long  — fallback if no direct generation column found;
               expects fuel_type + value structure
    3. Error — neither format could be inferred
    """
    try:
        df_raw = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if df_raw.empty:
        raise ValueError("Uploaded CSV is empty.")

    cols      = df_raw.columns.tolist()
    detection = detect_schema(cols)
    warnings  = list(detection.warnings)

    if detection.wind_col or detection.solar_col:
        df            = _parse_wide_csv(df_raw, detection)
        source_format = "wide_csv"
    else:
        # No direct generation column found — try long (AESO-style) format.
        # The timestamp may still be in detection.timestamp_col.
        df            = _parse_long_csv(df_raw, detection)
        source_format = "long_csv"

    return IngestResult(
        df=df,
        detection=detection,
        source_format=source_format,
        warnings=warnings,
        needs_review=detection.needs_review,
    )


# ─────────────────────────────────────────────────────────────────────────────
# XML parsers
# ─────────────────────────────────────────────────────────────────────────────

_IESO_NS = "http://www.ieso.ca/schema"


def _is_ieso_xml(root: ET.Element) -> bool:
    """Return True if the XML root uses the IESO schema namespace."""
    return _IESO_NS in root.tag or any(
        _IESO_NS in (v or "") for v in root.attrib.values()
    )


def _parse_ieso_xml_obj(raw: bytes) -> IngestResult:
    """
    Parse an IESO-style XML file (http://www.ieso.ca/schema).

    Structure: DailyData → HourlyData → FuelTotal
    IESO uses hour=24 to represent midnight of the next calendar day.
    """
    ns = {"ns": _IESO_NS}
    try:
        tree = ET.parse(io.BytesIO(raw))
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML: {exc}") from exc

    root = tree.getroot()
    rows: list[dict] = []

    for daily in root.findall(".//ns:DailyData", ns):
        day_el = daily.find("ns:Day", ns)
        if day_el is None or not day_el.text:
            continue
        date = day_el.text.strip()

        for hourly in daily.findall("ns:HourlyData", ns):
            hour_el = hourly.find("ns:Hour", ns)
            if hour_el is None or not hour_el.text:
                continue
            try:
                hour_int = int(hour_el.text.strip())
            except ValueError:
                continue

            # IESO hour 24 = midnight of the next day
            if hour_int == 24:
                dt     = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
                ts_str = dt.strftime("%Y-%m-%d 00")
            else:
                ts_str = f"{date} {hour_int:02d}"

            wind_val = solar_val = 0.0
            for fuel in hourly.findall("ns:FuelTotal", ns):
                ftype  = fuel.find("ns:Fuel",                  ns)
                output = fuel.find("ns:EnergyValue/ns:Output", ns)
                if ftype is None or output is None or output.text is None:
                    continue
                label = ftype.text.strip().upper()
                try:
                    val = float(output.text.strip())
                except ValueError:
                    continue
                if label == "WIND":
                    wind_val = val
                elif label == "SOLAR":
                    solar_val = val

            rows.append({"timestamp": ts_str, "Wind": wind_val, "Solar": solar_val})

    if not rows:
        raise ValueError(
            "IESO XML parsed successfully but contained no Wind/Solar hourly records. "
            "Verify the file contains DailyData → HourlyData → FuelTotal elements."
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = _normalise_timestamps(df["timestamp"])

    # Construct a synthetic SchemaDetection reflecting what was found
    detection = SchemaDetection(
        timestamp_col="timestamp",
        wind_col="Wind",
        solar_col="Solar",
        timestamp_score=1.0,
        wind_score=1.0,
        solar_score=1.0,
    )

    return IngestResult(df=df, detection=detection, source_format="ieso_xml")


def _strip_ns(tag: str) -> str:
    """Strip an XML namespace prefix — e.g. '{http://...}TagName' → 'TagName'."""
    return re.sub(r"\{[^}]+\}", "", tag)


def _flatten_element(elem: ET.Element, prefix: str = "") -> dict[str, str]:
    """
    Recursively flatten an XML element into a flat {field_name: value} dict.

    Tag names become keys (dot-separated for nested elements).
    Attributes and text content are both captured.
    Namespaces are stripped from tag names for readability.
    """
    local_tag = _strip_ns(elem.tag)
    key       = f"{prefix}.{local_tag}" if prefix else local_tag
    result:   dict[str, str] = {}

    for attr_name, attr_val in elem.attrib.items():
        result[f"{key}.{_strip_ns(attr_name)}"] = attr_val

    text = (elem.text or "").strip()
    if text:
        result[key] = text

    for child in elem:
        result.update(_flatten_element(child, prefix=key))

    return result


def _find_record_elements(root: ET.Element) -> list[ET.Element]:
    """
    Find the largest group of sibling elements sharing the same tag — these
    are the repeating data records in a generic XML file.

    Searches depth-first and returns the group with the most records.
    """
    from collections import defaultdict

    def collect(node: ET.Element) -> list[tuple[int, list[ET.Element]]]:
        groups: list[tuple[int, list[ET.Element]]] = []
        tag_groups: dict[str, list[ET.Element]] = defaultdict(list)
        for child in node:
            tag_groups[child.tag].append(child)
        for children in tag_groups.values():
            if len(children) > 1:
                groups.append((len(children), children))
            for child in children:
                groups.extend(collect(child))
        return groups

    groups = collect(root)
    if not groups:
        return []

    groups.sort(key=lambda x: x[0], reverse=True)
    return groups[0][1]


def _parse_generic_xml_obj(raw: bytes) -> IngestResult:
    """
    Parse a non-IESO XML file by finding repeating record elements, flattening
    each into a row, building a DataFrame, and applying schema detection.

    Raises ValueError if no timestamp or generation fields can be detected.
    """
    try:
        tree = ET.parse(io.BytesIO(raw))
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML: {exc}") from exc

    root    = tree.getroot()
    records = _find_record_elements(root)

    if not records:
        raise ValueError(
            "Cannot find repeating data records in this XML file. "
            "Generic XML must contain a series of sibling elements where each "
            "represents one time-period record."
        )

    rows   = [_flatten_element(elem) for elem in records]
    df_raw = pd.DataFrame(rows)

    if df_raw.empty:
        raise ValueError("Generic XML parsed successfully but produced an empty dataset.")

    cols      = df_raw.columns.tolist()
    detection = detect_schema(cols)
    warnings  = list(detection.warnings)

    if detection.timestamp_col is None:
        raise ValueError(
            f"Cannot find a timestamp field in generic XML. "
            f"Fields detected: {cols[:30]}. "
            f"Expected a field containing 'date', 'time', 'datetime', 'timestamp', etc."
        )

    if detection.wind_col is None and detection.solar_col is None:
        raise ValueError(
            f"Cannot find Wind or Solar generation fields in generic XML. "
            f"Fields detected: {cols[:30]}."
        )

    df = _parse_wide_csv(df_raw, detection)

    return IngestResult(
        df=df,
        detection=detection,
        source_format="generic_xml",
        warnings=warnings,
        needs_review=detection.needs_review,
    )


def _parse_xml_obj(file_obj) -> IngestResult:
    """
    Dispatch XML parsing to the IESO-specific or generic handler.

    Reads the entire file into bytes so the stream can be passed to either
    handler without rewinding issues.
    """
    try:
        raw = file_obj.read()
        if isinstance(raw, str):
            raw = raw.encode()
    except Exception as exc:
        raise ValueError(f"Cannot read XML file: {exc}") from exc

    # Detect namespace to choose the right parser
    try:
        root = ET.parse(io.BytesIO(raw)).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML: {exc}") from exc

    if _is_ieso_xml(root):
        return _parse_ieso_xml_obj(raw)
    else:
        return _parse_generic_xml_obj(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Final normalization applied to every ingest path
# ─────────────────────────────────────────────────────────────────────────────

def _finalize(result: IngestResult) -> IngestResult:
    """
    Apply final normalization to the DataFrame inside an IngestResult:
    - Ensure Wind and Solar columns exist (NaN if absent)
    - Ensure timestamp is tz-naive datetime64[ns] floored to hour
    - Deduplicate same-hour rows with mean aggregation
    - Sort ascending by timestamp
    - Trim to [timestamp, Wind, Solar] columns only
    """
    df = result.df

    for col in ("Wind", "Solar"):
        if col not in df.columns:
            df[col] = float("nan")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["timestamp"] = df["timestamp"].dt.floor("h")

    df = (
        df.groupby("timestamp", as_index=False)
          .agg({"Wind": "mean", "Solar": "mean"})
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    result.df = df[["timestamp", "Wind", "Solar"]]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def ingest_with_metadata(file_obj, file_format: str) -> IngestResult:
    """
    Ingest an uploaded file and return a rich IngestResult with detection metadata.

    Parameters
    ----------
    file_obj    : file-like object (e.g. UploadFile.file or open("f", "rb"))
    file_format : "csv" or "xml"

    Returns
    -------
    IngestResult
        .df            → canonical DataFrame[timestamp, Wind, Solar]
        .detection     → SchemaDetection with per-field scores and reasons
        .source_format → "wide_csv", "long_csv", "ieso_xml", or "generic_xml"
        .warnings      → all warnings from detection and parsing
        .needs_review  → True if any detections were ambiguous or low-confidence
    """
    if file_format == "csv":
        result = _parse_csv_obj(file_obj)
    elif file_format == "xml":
        result = _parse_xml_obj(file_obj)
    else:
        raise ValueError(
            f"Unsupported file_format {file_format!r}. Use 'csv' or 'xml'."
        )

    return _finalize(result)


def ingest(file_obj, file_format: str) -> pd.DataFrame:
    """
    Ingest an uploaded file and return the canonical normalized DataFrame.

    This is the backward-compatible entry point used by
    pipelines/user_pipeline.py. It has the same signature and return contract
    as before but now uses the scoring-based schema detector internally.

    Parameters
    ----------
    file_obj    : file-like object (e.g. UploadFile.file)
    file_format : "csv" or "xml"

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime64[ns] tz-naive hourly),
                 Wind (float64 or NaN),
                 Solar (float64 or NaN).
        One row per unique hour, sorted ascending. No duplicate timestamps.
    """
    return ingest_with_metadata(file_obj, file_format).df
