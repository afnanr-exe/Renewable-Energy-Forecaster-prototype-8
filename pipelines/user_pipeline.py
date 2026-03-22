import os
from typing import List

import pandas as pd

from adapters.user_adapter import ingest
from weather.weather_fetcher import geocode_city, fetch_weather


def _validate(df: pd.DataFrame) -> None:
    """Raise ValueError with a descriptive message if the normalized DataFrame fails any rule."""
    if df.empty:
        raise ValueError("Uploaded file contains no usable data.")

    min_year = int(df["timestamp"].dt.year.min())
    max_year = int(df["timestamp"].dt.year.max())
    if max_year - min_year > 10:
        raise ValueError(
            f"Uploaded data spans {max_year - min_year} years "
            f"({min_year}–{max_year}). Maximum allowed is 10 years."
        )

    # Check that at least one fuel type has real (non-NaN) values
    has_wind  = "Wind"  in df.columns and df["Wind"].notna().any()
    has_solar = "Solar" in df.columns and df["Solar"].notna().any()
    if not has_wind and not has_solar:
        raise ValueError(
            "File must contain at least one non-empty Wind or Solar column. "
            "Check that column names match known aliases "
            "(e.g. 'Wind', 'wind_mw', 'Solar', 'solar_mw', 'Fuel Type' + 'Volume', etc.)."
        )

    # Verify hourly interval (median step ≈ 3600 s)
    diffs = df["timestamp"].sort_values().diff().dropna().dt.total_seconds()
    if len(diffs) > 0:
        median_step = diffs.median()
        if abs(median_step - 3600) > 1:
            raise ValueError(
                f"Timestamps are not hourly (median interval = {median_step / 3600:.2f} h). "
                "Data must be at 1-hour intervals."
            )


def build_user_master(
    upload_mode: str,
    file_format: str,
    files: List,
    output_dir: str,
    city: str,
    timezone: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if upload_mode == "single":
        df = ingest(files[0].file, file_format)
        _validate(df)

    elif upload_mode == "multi":
        dfs = [ingest(f.file, file_format) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        # Re-aggregate after concat to collapse overlapping timestamps between files.
        # Using mean: if two files report the same hour it averages the readings;
        # for non-overlapping ranges (the normal case) there is nothing to average.
        agg_cols = {c: "mean" for c in ("Wind", "Solar") if c in df.columns}
        df = (
            df.groupby("timestamp", as_index=False)
              .agg(agg_cols)
              .sort_values("timestamp")
              .reset_index(drop=True)
        )
        _validate(df)

    else:
        raise ValueError("upload_mode must be 'single' or 'multi'.")

    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date   = df["timestamp"].max().strftime("%Y-%m-%d")

    lat, lon = geocode_city(city)
    weather_rows = fetch_weather(lat, lon, start_date, end_date, timezone=timezone)
    weather_df = pd.DataFrame(weather_rows)
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    merged = pd.merge(df, weather_df, on="timestamp", how="left")

    master_path = os.path.join(output_dir, "upload_master.csv")
    merged.to_csv(master_path, index=False)
    return master_path
