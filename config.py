"""
Pearls AQI Predictor - shared configuration.

Single source of truth for cities, API endpoints, feature/target settings,
and Hopsworks naming, so the feature pipeline, training pipeline, and
dashboard never drift from each other.
"""

from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Cities
# --------------------------------------------------------------------------
CITIES: dict[str, tuple[float, float]] = {
    "lahore":     (31.5497, 74.3436),
    "islamabad":  (33.6844, 73.0479),
    "rawalpindi": (33.5651, 73.0169),
    "karachi":    (24.8607, 67.0011),
    "peshawar":   (34.0151, 71.5249),
}
DEFAULT_CITY = "lahore"

# --------------------------------------------------------------------------
# History / forecast
# --------------------------------------------------------------------------
HISTORY_DAYS = 365            # backfill window (Open-Meteo air-quality archive is safe ~2yr)
FORECAST_HORIZONS_H = [24, 48, 72]   # predict AQI this many hours ahead

# --------------------------------------------------------------------------
# Open-Meteo endpoints (no API key required)
# --------------------------------------------------------------------------
AQ_VARS = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide",
           "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "wind_direction_10m", "surface_pressure", "precipitation"]

AQ_HISTORY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_FORECAST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# --------------------------------------------------------------------------
# Alerting
# --------------------------------------------------------------------------
HAZARD_AQI = 150   # US AQI "Unhealthy (sensitive groups)" and above

# --------------------------------------------------------------------------
# Local paths (fallback store + always-on local cache)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Hopsworks (feature store + model registry)
# --------------------------------------------------------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_ENABLED = bool(HOPSWORKS_API_KEY and HOPSWORKS_PROJECT)

FEATURE_GROUP_VERSION = 1
MODEL_REGISTRY_VERSION = 1


def feature_group_name(city: str) -> str:
    return f"aqi_{city}"


def model_name(city: str, horizon_h: int) -> str:
    return f"aqi_{city}_t{horizon_h}h"
