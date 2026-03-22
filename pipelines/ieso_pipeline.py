import os
import csv
from datetime import datetime, timedelta
from adapters.ieso_adapter import download_xml, parse_xml_to_rows
from weather.weather_fetcher import geocode_city, fetch_weather

def trim_last_2p5_years(rows):
    cutoff = datetime.now() - timedelta(days=913)
    cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
    trimmed = []
    for r in rows:
        dt = datetime.strptime(r["timestamp"], "%Y-%m-%d %H")
        if dt >= cutoff:
            trimmed.append(r)
    return trimmed

def build_ieso_master(xml_dir, output_dir, city, timezone):
    today = datetime.now()
    current_year = today.year
    years = [current_year - 3, current_year - 2, current_year - 1, current_year] if today.month < 7 else [current_year - 2, current_year - 1, current_year]

    all_rows = []
    for y in years:
        xml_path = download_xml(y, xml_dir)
        if xml_path:
            all_rows.extend(parse_xml_to_rows(xml_path))

    if not all_rows:
        raise RuntimeError("No IESO data parsed from XML files.")

    all_rows = trim_last_2p5_years(all_rows)
    all_rows.sort(key=lambda r: r["timestamp"])

    if not all_rows:
        raise RuntimeError("No IESO rows remain after trimming to last 2.5 years.")

    start_date = all_rows[0]["timestamp"].split()[0]
    end_date = all_rows[-1]["timestamp"].split()[0]

    lat, lon = geocode_city(city)
    weather_rows = fetch_weather(lat, lon, start_date, end_date, timezone=timezone)
    weather_map = {r["timestamp"]: r for r in weather_rows}

    merged = []
    for r in all_rows:
        ts = r["timestamp"]
        if ts in weather_map:
            merged.append({**r, **weather_map[ts]})

    if not merged:
        raise RuntimeError(
            f"IESO weather merge produced no rows. "
            f"Generation timestamps sample: {[r['timestamp'] for r in all_rows[:3]]}. "
            f"Weather timestamps sample: {list(weather_map.keys())[:3]}."
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ieso_master.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    return out_path
