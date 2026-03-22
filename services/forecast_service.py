"""
services/forecast_service.py

Load a no-lag linear model trained on weather features only, fetch a 48-hour
weather forecast from Open-Meteo, and produce hourly wind/solar predictions.

Why no-lag?  The lag model (with lag1 as a feature) achieves high historical
R² because the previous hour's actual value is a strong predictor of the
current hour.  But when used recursively for future forecasting each predicted
value becomes the next step's lag, causing two failure modes:
  • Solar drifts upward indefinitely (the intercept term accumulates).
  • Wind can collapse to 0 and get stuck (lag=0 → next lag=0, indefinitely).

The no-lag model uses only weather variables (windspeed, cloudcover,
shortwave_radiation, etc.) which are available from the forecast API and have
a direct physical relationship with generation.  shortwave_radiation = 0 at
night guarantees solar predictions are naturally near-zero overnight.
"""

import os

import joblib
import numpy as np

from weather.weather_fetcher import geocode_city_full, fetch_forecast_weather

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")

# Threshold below which shortwave_radiation (W/m²) means it is night-time.
# Open-Meteo returns exactly 0.0 for dark hours; the small buffer handles
# any rounding in edge-of-twilight hours.
_NIGHT_RADIATION_THRESHOLD = 5.0


def run_forecast(market: str, city: str) -> dict:
    """
    Produce a 48-hour generation forecast for the given market and city.

    Parameters
    ----------
    market : str   "ieso", "aeso", or "upload"
    city   : str   City name geocoded via Open-Meteo (e.g. "Goderich")

    Returns
    -------
    dict with keys:
        market        : str
        city          : str
        city_timezone : IANA timezone string (e.g. "America/Toronto")
        hours         : list of 48 dicts, each with:
                            hour      int  (1–48)
                            utc_iso   str  "2025-02-20T14:00"
                            wind_mw   float
                            solar_mw  float
    """
    label      = market.upper()
    wind_path  = os.path.join(MODELS_DIR, f"{label}_Wind_forecast.pkl")
    solar_path = os.path.join(MODELS_DIR, f"{label}_Solar_forecast.pkl")

    missing = [p for p in (wind_path, solar_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"No trained forecast model found for market '{market}'. "
            f"Run the {market.upper()} pipeline first to train the models."
        )

    wind_art  = joblib.load(wind_path)
    solar_art = joblib.load(solar_path)

    lat, lon, city_tz = geocode_city_full(city)
    forecast_rows     = fetch_forecast_weather(lat, lon)   # UTC ISO timestamps, 48 h

    hours = []
    for i, row in enumerate(forecast_rows):
        # ── Wind prediction ───────────────────────────────────────────────────
        X_wind = np.array([[float(row.get(f) or 0) for f in wind_art["features"]]])
        wind_pred = float(max(0.0, wind_art["model"].predict(X_wind)[0]))

        # ── Solar prediction ──────────────────────────────────────────────────
        # Hard physical constraint: if there is no solar radiation (nighttime),
        # generation must be zero regardless of what the model says.
        radiation = float(row.get("shortwave_radiation") or 0)
        if radiation < _NIGHT_RADIATION_THRESHOLD:
            solar_pred = 0.0
        else:
            X_solar = np.array([[float(row.get(f) or 0) for f in solar_art["features"]]])
            solar_pred = float(max(0.0, solar_art["model"].predict(X_solar)[0]))

        hours.append({
            "hour":                i + 1,
            "utc_iso":             row["utc_iso"],
            "wind_mw":             round(wind_pred,  1),
            "solar_mw":            round(solar_pred, 1),
            "temperature_2m":      row.get("temperature_2m"),
            "windspeed_10m":       row.get("windspeed_10m"),
            "cloudcover":          row.get("cloudcover"),
            "shortwave_radiation": row.get("shortwave_radiation"),
        })

    return {
        "market":        market,
        "city":          city,
        "city_timezone": city_tz,
        "lat":           round(lat, 5),
        "lon":           round(lon, 5),
        "wind_r2":       round(wind_art.get("r2", 0.0), 4),
        "solar_r2":      round(solar_art.get("r2", 0.0), 4),
        "hours":         hours,
    }
