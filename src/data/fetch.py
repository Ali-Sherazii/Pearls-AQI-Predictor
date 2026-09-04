"""
Raw data access: Open-Meteo air-quality + weather, for any configured city.

Two usage modes:
  - history:  fetch_raw_history(city, history_days)   -> backfill training data
  - serving:  fetch_recent_airquality / fetch_forecast_weather -> live forecasting

Both merge onto the union of AQ_VARS/WX_VARS so downstream feature engineering
(src/features/engineering.py) works identically on either.
"""

from __future__ import annotations
import time
from datetime import date, timedelta

import pandas as pd
import requests

from config import (
    CITIES, AQ_VARS, WX_VARS,
    AQ_HISTORY_URL, WX_HISTORY_URL, AQ_FORECAST_URL, WX_FORECAST_URL,
)


def _get_hourly(url: str, params: dict, tries: int = 4) -> pd.DataFrame:
    """GET an Open-Meteo endpoint and return its `hourly` block as a DataFrame."""
    last_exc = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            # network-level failures (read timeouts, connection resets) - the
            # original version only retried on a non-200 status code, so one
            # of these would crash the whole pipeline with zero retries.
            last_exc = exc
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            hourly = r.json()["hourly"]
            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            return df.set_index("time")
        last_exc = None
        # 429 = rate limited; back off and retry
        time.sleep(3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    r.raise_for_status()


def _latlon(city: str) -> tuple[float, float]:
    try:
        return CITIES[city]
    except KeyError:
        raise ValueError(f"Unknown city {city!r}; configured cities: {list(CITIES)}")


# --------------------------------------------------------------------------
# Historical (backfill)
# --------------------------------------------------------------------------
def fetch_air_quality_history(city: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = _latlon(city)
    return _get_hourly(AQ_HISTORY_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(AQ_VARS),
        "start_date": start, "end_date": end,
        "timezone": "auto",
    })


def fetch_weather_history(city: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = _latlon(city)
    return _get_hourly(WX_HISTORY_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(WX_VARS),
        "start_date": start, "end_date": end,
        "timezone": "auto",
    })


def fetch_raw_history(city: str, history_days: int) -> pd.DataFrame:
    """Fetch + merge air quality and weather into one hourly raw table."""
    end = date.today() - timedelta(days=5)      # ERA5 archive lags ~5 days
    start = end - timedelta(days=history_days)
    s, e = start.isoformat(), end.isoformat()

    aq = fetch_air_quality_history(city, s, e)
    wx = fetch_weather_history(city, s, e)

    raw = aq.join(wx, how="inner").sort_index()  # inner join: hours present in both
    raw = raw.dropna(subset=["us_aqi"])          # target must exist
    return raw


# --------------------------------------------------------------------------
# Serving (recent observations + forecast weather)
# --------------------------------------------------------------------------
def fetch_recent_airquality(city: str, past_days: int = 3, forecast_days: int = 1) -> pd.DataFrame:
    lat, lon = _latlon(city)
    return _get_hourly(AQ_FORECAST_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(AQ_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "auto",
    })


def fetch_forecast_weather(city: str, past_days: int = 3, forecast_days: int = 4) -> pd.DataFrame:
    lat, lon = _latlon(city)
    return _get_hourly(WX_FORECAST_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(WX_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "auto",
    })


def fetch_recent_raw(city: str, past_days: int = 10) -> pd.DataFrame:
    """Recent observed AQ + weather (not the lagged archive) - what the hourly
    feature pipeline writes into the store. Uses the same forecast-family
    endpoints as fetch_forecast_weather, just discarding the future part."""
    aq = fetch_recent_airquality(city, past_days=past_days, forecast_days=1)
    wx = fetch_forecast_weather(city, past_days=past_days, forecast_days=1)
    wx_cols = [c for c in WX_VARS if c in wx.columns]
    raw = aq.join(wx[wx_cols], how="inner").sort_index()
    return raw.dropna(subset=["us_aqi"])
