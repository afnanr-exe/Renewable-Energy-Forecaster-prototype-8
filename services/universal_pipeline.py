import os
import pandas as pd
import yaml
from fastapi import HTTPException

from models.regression_engine import run_both_models
from pipelines.ieso_pipeline import build_ieso_master
from pipelines.aeso_pipeline import build_aeso_master

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UniversalPipeline:
    def __init__(self):
        self.config_path = os.path.join(BASE_DIR, "config.yaml")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.output_base = os.path.join(BASE_DIR, "output")
        os.makedirs(self.output_base, exist_ok=True)

    def _get_paths(self, market: str):
        market_dir = os.path.join(self.output_base, market)
        os.makedirs(market_dir, exist_ok=True)
        return {"market_output_dir": market_dir}

    def _run_model_safe(
        self,
        csv_path: str,
        target: str,
        features: list[str],
        label: str,
    ) -> dict:
        """
        Run regression for one fuel type.  On success returns the normal
        run_both_models dict.  On failure (e.g. insufficient data for an
        upload-only fuel type) returns {"skipped": True, "reason": str(e)}
        so the other fuel type can still be reported.
        """
        try:
            return run_both_models(
                csv_path=csv_path,
                target=target,
                features=features,
                label=label,
            )
        except Exception as e:
            return {"skipped": True, "reason": str(e)}

    def run_market(
        self,
        market: str,
        city: str,
        upload_mode: str | None = None,
        file_format: str | None = None,
        files=None,
        timezone: str | None = None,
    ):
        try:
            return self._run_market_inner(
                market, city, upload_mode, file_format, files, timezone
            )
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error [{market}]: {e}")

    def _run_market_inner(
        self,
        market: str,
        city: str,
        upload_mode,
        file_format,
        files,
        timezone: str | None = None,
    ):
        # ---------------- UPLOAD MARKET ----------------
        if market == "upload":
            from pipelines.user_pipeline import build_user_master

            paths = self._get_paths("upload")
            market_output_dir = paths["market_output_dir"]

            master_path = build_user_master(
                upload_mode=upload_mode,
                file_format=file_format,
                files=files,
                output_dir=market_output_dir,
                city=city,
                timezone=timezone or "UTC",
            )

        else:
            # ---------------- IESO / AESO ----------------
            paths = self._get_paths(market)
            market_output_dir = paths["market_output_dir"]

            if market == "ieso":
                xml_dir  = os.path.join(BASE_DIR, self.config["markets"]["ieso"]["xml_dir"])
                tz       = self.config["markets"]["ieso"].get("timezone", "UTC")
                master_path = build_ieso_master(
                    xml_dir=xml_dir,
                    output_dir=market_output_dir,
                    city=city,
                    timezone=tz,
                )

            elif market == "aeso":
                input_dir = os.path.join(BASE_DIR, self.config["markets"]["aeso"]["csv_dir"])
                tz        = self.config["markets"]["aeso"].get("timezone", "America/Edmonton")
                master_path = build_aeso_master(
                    input_dir=input_dir,
                    output_dir=market_output_dir,
                    city=city,
                    timezone=tz,
                )

            else:
                raise ValueError(f"Unknown market: {market}")

        # ---------------- MODEL-READY CSVs ----------------
        df = pd.read_csv(master_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        wind_cols  = ["timestamp", "Wind",  "temperature_2m", "windspeed_10m", "winddirection_10m"]
        solar_cols = ["timestamp", "Solar", "temperature_2m", "cloudcover", "shortwave_radiation"]

        wind_features  = ["temperature_2m", "windspeed_10m", "winddirection_10m"]
        solar_features = ["temperature_2m", "cloudcover", "shortwave_radiation"]

        missing_wind  = [c for c in wind_cols  if c not in df.columns]
        missing_solar = [c for c in solar_cols if c not in df.columns]
        if missing_wind:
            raise ValueError(f"Master CSV missing columns for wind model: {missing_wind}")
        if missing_solar:
            raise ValueError(f"Master CSV missing columns for solar model: {missing_solar}")

        wind_df  = df.dropna(subset=["Wind"])[wind_cols]
        solar_df = df.dropna(subset=["Solar"])[solar_cols]

        wind_csv_path  = os.path.join(market_output_dir, "wind_model_data.csv")
        solar_csv_path = os.path.join(market_output_dir, "solar_model_data.csv")

        wind_df.to_csv(wind_csv_path,   index=False)
        solar_df.to_csv(solar_csv_path, index=False)

        # ---------------- RUN MODELS ----------------
        # _run_model_safe returns a skipped sentinel instead of raising if a
        # fuel type has insufficient data (e.g. wind-only or solar-only uploads).
        wind_results = self._run_model_safe(
            csv_path=wind_csv_path,
            target="Wind",
            features=wind_features,
            label=f"{market.upper()}_Wind",
        )

        solar_results = self._run_model_safe(
            csv_path=solar_csv_path,
            target="Solar",
            features=solar_features,
            label=f"{market.upper()}_Solar",
        )

        return {
            "market": market,
            "city": city,
            "wind":  wind_results,
            "solar": solar_results,
            "master_path": master_path,
            "wind_csv":    wind_csv_path,
            "solar_csv":   solar_csv_path,
        }
