import os
import csv
from adapters.aeso_adapter import load_aeso_generation, build_aeso_wind_solar
from weather.weather_fetcher import geocode_city, fetch_weather


def build_aeso_master(input_dir, output_dir, city, timezone):
    rows = load_aeso_generation(input_dir)
    if not rows:
        raise RuntimeError(f"No AESO rows loaded from '{input_dir}'. Check the csv_dir config path.")

    wind_rows, solar_rows = build_aeso_wind_solar(rows)
    if not wind_rows and not solar_rows:
        raise RuntimeError("AESO adapter found no WIND or SOLAR records in the CSV files.")

    wind_map = {r["timestamp"]: r["Wind"] for r in wind_rows}
    solar_map = {r["timestamp"]: r["Solar"] for r in solar_rows}

    # Use UNION so wind-only timestamps (e.g. nighttime, no solar) are kept.
    # Solar will be None for those hours; wind will be None for solar-only hours.
    all_timestamps = sorted(set(wind_map.keys()) | set(solar_map.keys()))

    aeso_rows = [
        {
            "timestamp": ts,
            "Wind": wind_map.get(ts),
            "Solar": solar_map.get(ts),
        }
        for ts in all_timestamps
    ]

    start_date = all_timestamps[0].split()[0]
    end_date = all_timestamps[-1].split()[0]

    lat, lon = geocode_city(city)
    weather_rows = fetch_weather(lat, lon, start_date, end_date, timezone=timezone)
    weather_map = {r["timestamp"]: r for r in weather_rows}

    merged = []
    for r in aeso_rows:
        ts = r["timestamp"]
        if ts in weather_map:
            merged.append({**r, **weather_map[ts]})

    if not merged:
        raise RuntimeError(
            f"AESO weather merge produced no rows. "
            f"Timezone is '{timezone}' — AESO data is MST (UTC-7). "
            f"Generation timestamps sample: {all_timestamps[:3]}. "
            f"Weather timestamps sample: {list(weather_map.keys())[:3]}."
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "aeso_master.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    return out_path
