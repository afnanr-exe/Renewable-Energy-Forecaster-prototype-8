import os
from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles

from services.universal_pipeline import UniversalPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
pipeline = UniversalPipeline()


class UploadMode(str, Enum):
    single = "single"
    multi  = "multi"


class MarketFormat(str, Enum):
    aeso = "aeso"
    ieso = "ieso"


class Province(str, Enum):
    alberta = "alberta"
    ontario = "ontario"
    other   = "other"


DEFAULT_IESO_CITY = "Goderich"
DEFAULT_AESO_CITY = "Red Deer"

# Maps each province to the timezone used when fetching weather for uploaded data.
# Must match the timezone the source data was recorded in so timestamps align.
PROVINCE_TIMEZONE: dict[Province, str] = {
    Province.alberta: "America/Edmonton",   # AESO data is MST (UTC-7)
    Province.ontario: "UTC",                # IESO data timestamps are UTC
    Province.other:   "UTC",               # safe default; user bears responsibility
}


def to_url_path(abs_path: str) -> str:
    """Convert an absolute path under BASE_DIR to a /output/... URL the browser can fetch."""
    if not abs_path:
        return abs_path
    try:
        rel = os.path.relpath(abs_path, BASE_DIR).replace("\\", "/")
        return f"/{rel}"
    except ValueError:
        return abs_path  # different drive on Windows — return as-is


def convert_paths(result: dict) -> dict:
    """
    Rewrite all absolute plot/CSV paths in the pipeline result to
    browser-accessible URL paths.

    Guards against None and {"skipped": True} model results so that
    a single-fuel-type upload (where one model is skipped) does not crash.
    """
    for fuel in ("wind", "solar"):
        fuel_data = result.get(fuel)
        # Skip None (not returned) or skipped-model sentinel
        if not fuel_data or fuel_data.get("skipped"):
            continue
        for model in ("linear", "polynomial"):
            m = fuel_data.get(model) or {}
            for key in ("scatter_plot", "timeseries_plot"):
                if key in m:
                    m[key] = to_url_path(m[key])
    for key in ("master_path", "wind_csv", "solar_csv"):
        if key in result:
            result[key] = to_url_path(result[key])
    return result


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/run-ieso")
def run_ieso():
    return convert_paths(pipeline.run_market("ieso", city=DEFAULT_IESO_CITY))


@app.get("/run-aeso")
def run_aeso():
    return convert_paths(pipeline.run_market("aeso", city=DEFAULT_AESO_CITY))


@app.post("/run-upload")
def run_upload(
    upload_mode:   UploadMode    = Form(...),
    market_format: MarketFormat  = Form(...),
    province:      Province      = Form(...),
    other_city:    Optional[str] = Form(None),
    files:         List[UploadFile] = File(...),
):
    file_format = "csv" if market_format == MarketFormat.aeso else "xml"

    if province == Province.ontario:
        final_city = DEFAULT_IESO_CITY
    elif province == Province.alberta:
        final_city = DEFAULT_AESO_CITY
    else:
        if not other_city:
            raise HTTPException(
                status_code=400,
                detail="Please provide a city name when selecting 'other'.",
            )
        final_city = other_city

    if upload_mode == UploadMode.single and len(files) != 1:
        raise HTTPException(status_code=400, detail="Single-file mode requires exactly 1 file.")
    if upload_mode == UploadMode.multi and len(files) < 2:
        raise HTTPException(status_code=400, detail="Multi-file mode requires at least 2 files.")

    tz = PROVINCE_TIMEZONE[province]

    # ValueError/RuntimeError are caught inside run_market and returned as HTTPException
    return convert_paths(pipeline.run_market(
        "upload",
        city=final_city,
        upload_mode=upload_mode.value,
        file_format=file_format,
        files=files,
        timezone=tz,
    ))


@app.get("/run-forecast")
def run_forecast_endpoint(market: str, city: str):
    from services.forecast_service import run_forecast
    try:
        return run_forecast(market, city)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {e}")


# ── Static file mounts (must come after all API route definitions) ─────────────

_output_dir  = os.path.join(BASE_DIR, "output")
_plots_dir   = os.path.join(_output_dir, "plots")
_frontend_dir = os.path.join(BASE_DIR, "frontend")

os.makedirs(_plots_dir,    exist_ok=True)
os.makedirs(_frontend_dir, exist_ok=True)

# Serve output/ at /output — plots and CSVs
app.mount("/output", StaticFiles(directory=_output_dir),  name="output")

# Serve frontend/ at / — must be last so API routes above take priority
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
