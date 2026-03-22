import os
import csv
from collections import defaultdict
from datetime import datetime

def load_aeso_generation(input_dir: str):
    all_rows = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows.extend(list(reader))
    return all_rows

def build_aeso_wind_solar(rows):
    wind_data = defaultdict(list)
    solar_data = defaultdict(list)

    for r in rows:
        fuel = r.get("Fuel Type", "").strip().upper()
        ts_raw = r.get("Date (MST)", "").strip()
        vol_raw = r.get("Volume", "").strip()
        if fuel not in {"WIND", "SOLAR"} or not ts_raw or not vol_raw:
            continue
        try:
            dt = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
            ts = dt.strftime("%Y-%m-%d %H")
            vol = float(vol_raw)
        except Exception:
            continue
        if fuel == "WIND":
            wind_data[ts].append(vol)
        else:
            solar_data[ts].append(vol)

    wind_rows = [{"timestamp": ts, "Wind": round(sum(vals), 2)} for ts, vals in wind_data.items()]
    solar_rows = [{"timestamp": ts, "Solar": round(sum(vals), 2)} for ts, vals in solar_data.items()]
    wind_rows.sort(key=lambda r: r["timestamp"])
    solar_rows.sort(key=lambda r: r["timestamp"])
    return wind_rows, solar_rows
