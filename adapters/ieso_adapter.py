import os
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

BASE_URL = "https://reports-public.ieso.ca/public/GenOutputbyFuelHourly/"

def get_latest_version_file(year):
    r = requests.get(BASE_URL, timeout=15)
    r.raise_for_status()
    html = r.text
    pattern_v = rf"PUB_GenOutputbyFuelHourly_{year}_v(\d+)\.xml"
    matches_v = re.findall(pattern_v, html)
    if matches_v:
        latest_v = max(int(v) for v in matches_v)
        filename = f"PUB_GenOutputbyFuelHourly_{year}_v{latest_v}.xml"
        return BASE_URL + filename, filename
    pattern_base = rf"PUB_GenOutputbyFuelHourly_{year}\.xml"
    if re.search(pattern_base, html):
        filename = f"PUB_GenOutputbyFuelHourly_{year}.xml"
        return BASE_URL + filename, filename
    return None, None

def download_xml(year, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    url, filename = get_latest_version_file(year)
    if not url:
        return None
    out_path = os.path.join(out_folder, filename)
    if os.path.exists(out_path):
        return out_path
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(r.content)
    return out_path

def parse_xml_to_rows(xml_path):
    ns = {'ns': 'http://www.ieso.ca/schema'}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for daily_data in root.findall('.//ns:DailyData', ns):
        date = daily_data.find('ns:Day', ns).text.strip()
        for hourly in daily_data.findall('ns:HourlyData', ns):
            hour_raw = hourly.find('ns:Hour', ns).text.strip()
            try:
                hour_int = int(hour_raw)
            except Exception:
                continue
            if hour_int == 24:
                dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
                ts = dt.strftime("%Y-%m-%d 00")
            else:
                ts = f"{date} {hour_int:02d}"

            wind_val, solar_val = "0", "0"
            for fuel in hourly.findall('ns:FuelTotal', ns):
                ftype = fuel.find('ns:Fuel', ns)
                output = fuel.find('ns:EnergyValue/ns:Output', ns)
                if ftype is None or output is None or output.text is None:
                    continue
                fuel_name = ftype.text.strip().upper()
                if fuel_name == "WIND":
                    wind_val = output.text.strip()
                elif fuel_name == "SOLAR":
                    solar_val = output.text.strip()
            rows.append({"timestamp": ts, "Wind": float(wind_val), "Solar": float(solar_val)})
    rows.sort(key=lambda r: r["timestamp"])
    return rows
