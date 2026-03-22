"""
adapters/schema_detector.py

Scoring-based schema detection for uploaded renewable-energy datasets.

Instead of exact-match alias lookups, this module applies a multi-stage
scoring pipeline that handles real-world messy column names such as:

    "Wind Generation (MW)"   →  wind
    "WIND-GEN-HE"            →  wind
    "solar_output_mw"        →  solar
    "PV Power"               →  solar
    "Hour Ending"            →  timestamp
    "Settlement Timestamp"   →  timestamp

Public API
----------
normalize_field_name(name)   → str
score_field(name, target)    → FieldScore
detect_schema(columns)       → SchemaDetection
classify_fuel_label(label)   → "WIND" | "SOLAR" | None
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Primary identifiers — strong signal for each category
_WIND_PRIMARY:      frozenset[str] = frozenset({"wind", "wnd"})
_SOLAR_PRIMARY:     frozenset[str] = frozenset({"solar", "pv", "photovoltaic", "solr"})
_TIMESTAMP_PRIMARY: frozenset[str] = frozenset({"timestamp", "datetime", "date", "time", "dt"})

# Generation-context tokens — boost score when found with a primary keyword
_GENERATION_CONTEXT: frozenset[str] = frozenset({
    "generation", "gen", "output", "power", "mw", "mwh", "gwh",
    "energy", "total", "actual", "metered", "measured",
    "renewable", "produced", "volume", "amount",
})

# Meteorological tokens — suppress score when found with wind/solar keyword
# and NO generation context (likely a weather variable, not generation)
_WIND_WEATHER_TOKENS: frozenset[str] = frozenset({
    "speed", "direction", "gust", "velocity", "bearing",
    "mph", "kph", "knot",
})
_SOLAR_WEATHER_TOKENS: frozenset[str] = frozenset({
    "irradiance", "radiation", "dni", "ghi", "dhi",
    "angle", "azimuth", "zenith", "albedo",
})

# Timestamp-context tokens — boost score when found with a timestamp keyword
_TIMESTAMP_CONTEXT: frozenset[str] = frozenset({
    "utc", "mst", "mpt", "est", "pst", "et", "local",
    "interval", "period", "start", "end", "begin",
    "settlement", "hour", "he", "hourending",
})

# Standalone tokens that on their own are strong timestamp signals
_TIMESTAMP_STANDALONE: frozenset[str] = frozenset({
    "timestamp", "datetime", "hour", "he", "interval", "settlement", "period",
})

# Ambiguity penalties — reduce score when these appear (likely not actual generation)
_AMBIGUITY_PENALTIES: dict[str, float] = {
    "forecast":    0.25,
    "predicted":   0.25,
    "projected":   0.25,
    "capacity":    0.30,
    "rated":       0.30,
    "installed":   0.30,
    "nameplate":   0.30,
    "target":      0.20,
    "potential":   0.20,
    "available":   0.15,
    "curtailed":   0.15,
    "curtailment": 0.20,
    "plan":        0.15,
    "planned":     0.15,
}

# Actuality tokens — boost score (confirms real measured generation)
_ACTUAL_TOKENS: frozenset[str] = frozenset({
    "actual", "realtime", "metered", "measured",
    "reported", "historic", "historical", "observed",
})


# ─────────────────────────────────────────────────────────────────────────────
# Exact alias sets
#
# Matching is done on the *normalized* field name (after normalize_field_name).
# A hit here gives score = 1.0 immediately.
# ─────────────────────────────────────────────────────────────────────────────

_WIND_EXACT: frozenset[str] = frozenset({
    "wind", "wnd",
    "wind mw", "windmw",
    "wind generation", "windgeneration",
    "wind output", "windoutput",
    "wind energy", "windenergy",
    "wind power", "windpower",
    "wind gen", "windgen",
    "wind generation mw", "wind output mw", "wind mwh",
    "wind total", "renewable wind", "wind actual",
    "wind he", "wind gen he", "windgenhe",
})

_SOLAR_EXACT: frozenset[str] = frozenset({
    "solar", "pv", "photovoltaic",
    "solar mw", "solarmw",
    "solar generation", "solargeneration",
    "solar output", "solaroutput",
    "solar energy", "solarenergy",
    "solar power", "solarpower",
    "solar gen", "solargen",
    "solar generation mw", "solar output mw", "solar mwh",
    "pv output", "pv power", "pv generation", "pv mw",
    "photovoltaic power", "photovoltaic output", "photovoltaic generation",
    "renewable solar", "solar total", "solar actual",
})

_TIMESTAMP_EXACT: frozenset[str] = frozenset({
    "timestamp", "datetime", "date", "time", "date time", "dt",
    "datetime utc", "date utc", "time utc",
    "date mst", "date mpt", "date est", "date et", "date pst",
    "date time mst", "date time utc",
    "interval start", "interval end", "interval datetime",
    "settlement datetime", "settlement date", "settlement timestamp",
    "hour", "hour ending", "he", "hour ending he",
    "period", "period start", "period end",
    "local time", "local datetime",
    "date hr", "date hour",
})

# Confidence thresholds
_ACCEPT_THRESHOLD = 0.50   # minimum score to consider a field detected
_REVIEW_THRESHOLD = 0.70   # below this → add a needs_review warning
_AMBIGUITY_GAP    = 0.15   # if top-2 within this gap → ambiguous warning


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldScore:
    """
    Scoring result for one (column_name, target_type) pair.

    Attributes
    ----------
    name       : original column name as it appears in the file
    target     : "wind", "solar", or "timestamp"
    score      : float in [0, 1] — higher means more confident
    reasons    : human-readable strings explaining each score contribution
    normalized : the normalized form of `name` used during matching
    """
    name:       str
    target:     str
    score:      float
    reasons:    list[str] = field(default_factory=list)
    normalized: str = ""


@dataclass
class SchemaDetection:
    """
    Full schema detection result returned by detect_schema().

    Attributes
    ----------
    timestamp_col, wind_col, solar_col
        Best matching column name for each target, or None if not found.
    timestamp_score, wind_score, solar_score
        Confidence score for each detected assignment.
    warnings
        Human-readable strings for ambiguous or low-confidence detections.
    needs_review
        True when any detection is ambiguous or below _REVIEW_THRESHOLD.
    all_scores
        Every FieldScore computed — useful for debugging or display.
    """
    timestamp_col:   Optional[str] = None
    wind_col:        Optional[str] = None
    solar_col:       Optional[str] = None
    timestamp_score: float = 0.0
    wind_score:      float = 0.0
    solar_score:     float = 0.0
    warnings:        list[str] = field(default_factory=list)
    needs_review:    bool = False
    all_scores:      list[FieldScore] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_field_name(name: str) -> str:
    """
    Normalize a column/field name to a canonical lowercase string suitable
    for keyword-based matching.

    Processing steps
    ----------------
    1. Unicode NFC normalization
    2. Lowercase
    3. Strip parenthetical/bracketed suffixes — e.g. "(MW)", "[MWh]", "(MST)"
    4. Replace separators (_, -, /, \\, |) with spaces
    5. Remove any remaining non-word, non-space characters
    6. Collapse multiple spaces; strip leading/trailing whitespace

    Examples
    --------
    >>> normalize_field_name("Wind Generation (MW)")
    'wind generation'
    >>> normalize_field_name("WIND-GEN-HE")
    'wind gen he'
    >>> normalize_field_name("solar_output_mw")
    'solar output mw'
    >>> normalize_field_name("Date (MST)")
    'date'
    >>> normalize_field_name("Hour Ending")
    'hour ending'
    >>> normalize_field_name("PV Power [kW]")
    'pv power'
    >>> normalize_field_name("Settlement Timestamp")
    'settlement timestamp'
    """
    if not isinstance(name, str):
        name = str(name)

    # Unicode NFC
    name = unicodedata.normalize("NFC", name)

    # Lowercase
    name = name.lower()

    # Remove parenthetical/bracketed content (units, timezone codes, numbers)
    # Matches e.g. "(MW)", "[MWh]", "(MST)", "(2023)", "(m/s)"
    name = re.sub(r"[\(\[]\s*[\w\s°/\.\-\+%]+?\s*[\)\]]", " ", name)

    # Replace common separators with spaces
    name = re.sub(r"[_\-/\\|]", " ", name)

    # Remove any remaining non-word, non-space characters
    name = re.sub(r"[^\w\s]", " ", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


def _token_set(normalized: str) -> set[str]:
    """Return the set of individual word tokens from a normalized string."""
    return set(normalized.split())


# ─────────────────────────────────────────────────────────────────────────────
# Per-target scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_wind(normalized: str, tokens: set[str]) -> tuple[float, list[str]]:
    reasons: list[str] = []

    # Stage 1: exact alias match
    if normalized in _WIND_EXACT:
        reasons.append("exact alias match")
        return 1.0, reasons

    # Stage 2: primary wind keyword must be present
    primary_hit = tokens & _WIND_PRIMARY
    if not primary_hit:
        return 0.0, ["no wind keyword found"]

    reasons.append(f"wind keyword present: {primary_hit}")
    score = 0.65

    gen_ctx     = tokens & _GENERATION_CONTEXT
    weather_ctx = tokens & _WIND_WEATHER_TOKENS

    if gen_ctx and not weather_ctx:
        # Clean generation signal — boost proportional to number of context tokens
        boost = min(0.30, len(gen_ctx) * 0.10)
        score += boost
        reasons.append(f"+{boost:.0%} generation context tokens: {gen_ctx}")

    elif weather_ctx and not gen_ctx:
        # Looks like a weather/met variable (e.g. "wind speed", "wind direction")
        score = 0.10
        reasons.append(
            f"suppressed to 0.10: weather tokens {weather_ctx} "
            f"without generation context (likely a meteorological variable)"
        )

    elif gen_ctx and weather_ctx:
        # Both signals present — mild boost, flag ambiguity
        score += 0.05
        reasons.append(
            "mild +5% boost: both generation and weather tokens present "
            f"(gen={gen_ctx}, weather={weather_ctx})"
        )

    return score, reasons


def _score_solar(normalized: str, tokens: set[str]) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if normalized in _SOLAR_EXACT:
        reasons.append("exact alias match")
        return 1.0, reasons

    primary_hit = tokens & _SOLAR_PRIMARY
    if not primary_hit:
        return 0.0, ["no solar keyword found"]

    reasons.append(f"solar keyword present: {primary_hit}")
    score = 0.65

    gen_ctx     = tokens & _GENERATION_CONTEXT
    weather_ctx = tokens & _SOLAR_WEATHER_TOKENS

    if gen_ctx and not weather_ctx:
        boost = min(0.30, len(gen_ctx) * 0.10)
        score += boost
        reasons.append(f"+{boost:.0%} generation context tokens: {gen_ctx}")

    elif weather_ctx and not gen_ctx:
        score = 0.10
        reasons.append(
            f"suppressed to 0.10: weather tokens {weather_ctx} "
            f"without generation context (likely a meteorological variable)"
        )

    elif gen_ctx and weather_ctx:
        score += 0.05
        reasons.append(
            "mild +5% boost: both generation and weather tokens present "
            f"(gen={gen_ctx}, weather={weather_ctx})"
        )

    return score, reasons


def _score_timestamp(normalized: str, tokens: set[str]) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if normalized in _TIMESTAMP_EXACT:
        reasons.append("exact alias match")
        return 1.0, reasons

    primary_hit = tokens & _TIMESTAMP_PRIMARY
    if primary_hit:
        reasons.append(f"timestamp keyword present: {primary_hit}")
        score = 0.70
        ctx = tokens & _TIMESTAMP_CONTEXT
        if ctx:
            boost = min(0.25, len(ctx) * 0.08)
            score += boost
            reasons.append(f"+{boost:.0%} timestamp context tokens: {ctx}")
        return score, reasons

    # Standalone tokens that strongly imply a timestamp column even without
    # the explicit "date" / "time" keyword
    standalone = tokens & _TIMESTAMP_STANDALONE
    if standalone:
        reasons.append(f"standalone timestamp token: {standalone}")
        return 0.60, reasons

    return 0.0, ["no timestamp keyword found"]


# ─────────────────────────────────────────────────────────────────────────────
# Public scoring API
# ─────────────────────────────────────────────────────────────────────────────

def score_field(name: str, target: str) -> FieldScore:
    """
    Score how likely the column named `name` is to be a `target` column.

    Parameters
    ----------
    name   : raw column name, e.g. "Wind Generation (MW)"
    target : one of "wind", "solar", "timestamp"

    Returns
    -------
    FieldScore
        .score   ∈ [0, 1]
        .reasons list of strings explaining each contribution

    Score guide
    -----------
    1.0         Exact alias match — very high confidence
    0.85–0.95   Primary keyword + strong generation context
    0.65–0.84   Primary keyword + some context
    0.50–0.64   Primary keyword alone — detected but flagged for review
    < 0.50      Ambiguous or unrelated — rejected by detect_schema()
    """
    normalized = normalize_field_name(name)
    tokens     = _token_set(normalized)

    if target == "wind":
        score, reasons = _score_wind(normalized, tokens)
    elif target == "solar":
        score, reasons = _score_solar(normalized, tokens)
    elif target == "timestamp":
        score, reasons = _score_timestamp(normalized, tokens)
    else:
        raise ValueError(
            f"Unknown target {target!r}. Expected 'wind', 'solar', or 'timestamp'."
        )

    # Apply ambiguity penalties
    for tok, penalty in _AMBIGUITY_PENALTIES.items():
        if tok in tokens:
            score = max(0.0, score - penalty)
            reasons.append(f"-{penalty:.0%} penalty: ambiguous token '{tok}'")

    # Apply actuality boost
    actual_hit = tokens & _ACTUAL_TOKENS
    if actual_hit:
        boost = 0.10
        score = min(1.0, score + boost)
        reasons.append(f"+{boost:.0%} boost: actuality token {actual_hit}")

    return FieldScore(
        name=name,
        target=target,
        score=round(score, 4),
        reasons=reasons,
        normalized=normalized,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_schema(columns: list[str]) -> SchemaDetection:
    """
    Identify the best timestamp, wind, and solar columns from a list of names.

    Parameters
    ----------
    columns : raw column name strings in original case

    Returns
    -------
    SchemaDetection

    Algorithm
    ---------
    1. Score every (column, target) pair via score_field().
    2. For each target, pick the highest-scoring column if score ≥ ACCEPT_THRESHOLD.
    3. Flag ambiguity when top-2 scores are within AMBIGUITY_GAP of each other.
    4. Prevent the same column from being used for two targets
       (priority: timestamp > wind > solar).
    5. Populate needs_review and warnings accordingly.
    """
    result     = SchemaDetection()
    all_scores: list[FieldScore] = []

    targets = ("timestamp", "wind", "solar")
    scored: dict[str, list[FieldScore]] = {t: [] for t in targets}

    for col in columns:
        for t in targets:
            fs = score_field(col, t)
            scored[t].append(fs)
            all_scores.append(fs)

    result.all_scores = all_scores

    # Select best candidate for each target
    best_per_target: dict[str, Optional[FieldScore]] = {}

    for t in targets:
        ranked = sorted(scored[t], key=lambda x: x.score, reverse=True)

        if not ranked or ranked[0].score < _ACCEPT_THRESHOLD:
            best_per_target[t] = None
            continue

        best   = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        # Ambiguity: two candidates within AMBIGUITY_GAP of each other
        if (
            second is not None
            and second.score >= _ACCEPT_THRESHOLD
            and (best.score - second.score) < _AMBIGUITY_GAP
        ):
            result.warnings.append(
                f"Ambiguous {t} detection: "
                f"'{best.name}' (score={best.score:.2f}) vs "
                f"'{second.name}' (score={second.score:.2f}). "
                f"Chose '{best.name}'."
            )
            result.needs_review = True

        # Low-confidence detection
        if best.score < _REVIEW_THRESHOLD:
            result.warnings.append(
                f"Low-confidence {t}: '{best.name}' (score={best.score:.2f}). "
                f"Please verify this is correct."
            )
            result.needs_review = True

        best_per_target[t] = best

    # Assign to result, preventing column reuse (timestamp takes priority)
    assigned: set[str] = set()

    for t in targets:
        candidate = best_per_target.get(t)
        if candidate is None:
            continue

        if candidate.name in assigned:
            result.warnings.append(
                f"Column '{candidate.name}' already assigned to another target; "
                f"cannot reuse for {t}. Skipping."
            )
            result.needs_review = True
            continue

        assigned.add(candidate.name)

        if t == "timestamp":
            result.timestamp_col   = candidate.name
            result.timestamp_score = candidate.score
        elif t == "wind":
            result.wind_col   = candidate.name
            result.wind_score = candidate.score
        elif t == "solar":
            result.solar_col   = candidate.name
            result.solar_score = candidate.score

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fuel-label classifier (used for long-format CSV row-level matching)
# ─────────────────────────────────────────────────────────────────────────────

def classify_fuel_label(label: str) -> Optional[str]:
    """
    Classify a fuel-type label string as "WIND", "SOLAR", or None.

    Uses the same normalization and keyword matching as detect_schema() so that
    labels like "Wind Turbine", "WIND-POWER", "photovoltaic", "PV", and
    "Solar Generation" are all correctly classified.

    Parameters
    ----------
    label : a single fuel-type label value from the fuel_type column

    Returns
    -------
    "WIND" | "SOLAR" | None
    """
    normalized = normalize_field_name(label)
    tokens     = _token_set(normalized)

    wind_score,  _ = _score_wind(normalized, tokens)
    solar_score, _ = _score_solar(normalized, tokens)

    if wind_score >= _ACCEPT_THRESHOLD and wind_score >= solar_score:
        return "WIND"
    if solar_score >= _ACCEPT_THRESHOLD:
        return "SOLAR"
    return None
