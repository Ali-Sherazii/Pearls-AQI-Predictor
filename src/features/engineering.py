"""
Feature engineering — the ONLY place this logic lives.

Backfill, the hourly feature pipeline, training, and live serving (the
dashboard) all import from here, so train/serve skew is impossible by
construction. There used to be three copies of this logic (features.py,
predict.py, train_final.py); this module replaces all of them.
"""

from __future__ import annotations
from datetime import timedelta

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS_H, WX_VARS


# --------------------------------------------------------------------------
# Present-time features (hour/day/month, cyclical encodings, lags, rolling,
# change rate). Works on any contiguous hourly slice - backfill and the
# hourly pipeline both call this on whatever raw window they have.
# --------------------------------------------------------------------------
def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    t = df.index

    # time-based
    df["hour"] = t.hour
    df["day"] = t.day
    df["month"] = t.month
    df["day_of_week"] = t.dayofweek
    df["is_weekend"] = (t.dayofweek >= 5).astype(int)

    # cyclical encodings (help linear models; harmless for trees)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # wind direction is circular (0deg == 360deg) - as a raw degree value a
    # model sees north and "north minus one degree" as maximally different.
    if "wind_direction_10m" in df.columns:
        rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)

    # lags + rolling on the pollutants that actually drive AQI
    for col in ["us_aqi", "pm2_5", "pm10"]:
        df[f"{col}_lag1"] = df[col].shift(1)      # 1h ago
        df[f"{col}_lag24"] = df[col].shift(24)    # same hour yesterday
        df[f"{col}_roll6"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll24"] = df[col].rolling(24, min_periods=1).mean()

    # finer-grained recent AQI history and volatility - the original lag1/
    # lag24 pair skips everything in between, which matters most at 24h/48h
    for lag in (2, 3, 6, 12):
        df[f"us_aqi_lag{lag}"] = df["us_aqi"].shift(lag)
    df["us_aqi_rollstd24"] = df["us_aqi"].rolling(24, min_periods=2).std()

    # light touch on the other pollutants - just enough temporal context to
    # let the model use them as more than a single noisy instantaneous reading
    for col in ["carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag24"] = df[col].shift(24)

    # derived: AQI change rate
    df["aqi_change_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_24h"] = df["us_aqi"].diff(24)

    # a pressure drop often precedes a front moving through and dispersing
    # (or trapping) pollution - a leading indicator persistence can't see
    if "surface_pressure" in df.columns:
        df["pressure_change_3h"] = df["surface_pressure"].diff(3)

    return df


# --------------------------------------------------------------------------
# Targets: US AQI `h` hours into the future, one column per horizon.
# --------------------------------------------------------------------------
def build_targets(df: pd.DataFrame, horizons=FORECAST_HORIZONS_H) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"target_aqi_t{h}h"] = out["us_aqi"].shift(-h)
    return out


# --------------------------------------------------------------------------
# Future weather (training): shift the historical weather series so row t
# carries the weather that actually occurred at t+h, plus cumulative
# rain / mean wind between t+1..t+h. Used to build the training table, where
# we have the full historical series to shift.
# --------------------------------------------------------------------------
def build_future_weather(df: pd.DataFrame, h: int) -> pd.DataFrame:
    fut = pd.DataFrame(index=df.index)
    for w in WX_VARS:
        if w in df.columns:
            fut[f"{w}_t{h}h"] = df[w].shift(-h)
    if "wind_direction_10m" in df.columns:
        # replace the raw future degree value with its cyclic encoding, same
        # reasoning as engineer_features's present-time wind_dir_sin/cos
        rad = np.deg2rad(df["wind_direction_10m"].shift(-h))
        fut[f"wind_dir_sin_t{h}h"] = np.sin(rad)
        fut[f"wind_dir_cos_t{h}h"] = np.cos(rad)
        fut = fut.drop(columns=[f"wind_direction_10m_t{h}h"])
    if "precipitation" in df.columns:
        fut[f"precip_sum_next{h}h"] = df["precipitation"].rolling(h).sum().shift(-h)
    if "wind_speed_10m" in df.columns:
        fut[f"wind_mean_next{h}h"] = df["wind_speed_10m"].rolling(h).mean().shift(-h)
    return fut


# --------------------------------------------------------------------------
# Future weather (serving): same feature names as build_future_weather, but
# computed from a FORECAST weather frame (fcw) relative to a single "now"
# timestamp T, since serving doesn't have h more hours of real history to
# shift - it has to look up the forecast instead.
# --------------------------------------------------------------------------
def future_weather_at(fcw: pd.DataFrame, T: pd.Timestamp, h: int) -> dict:
    tgt = T + timedelta(hours=h)
    window = fcw.loc[(fcw.index > T) & (fcw.index <= tgt)]   # hours T+1..T+h
    feat = {}
    for w in WX_VARS:
        if w in fcw.columns:
            feat[f"{w}_t{h}h"] = fcw.at[tgt, w] if tgt in fcw.index else np.nan
    if "wind_direction_10m" in fcw.columns:
        wd = feat.pop(f"wind_direction_10m_t{h}h", np.nan)
        rad = np.deg2rad(wd) if pd.notna(wd) else np.nan
        feat[f"wind_dir_sin_t{h}h"] = np.sin(rad) if pd.notna(rad) else np.nan
        feat[f"wind_dir_cos_t{h}h"] = np.cos(rad) if pd.notna(rad) else np.nan
    if "precipitation" in fcw.columns:
        feat[f"precip_sum_next{h}h"] = window["precipitation"].sum()
    if "wind_speed_10m" in fcw.columns:
        feat[f"wind_mean_next{h}h"] = window["wind_speed_10m"].mean()
    return feat


# --------------------------------------------------------------------------
# Full feature table: present features + targets (targets are NaN for rows
# too recent to have a known future outcome yet - that's expected, both the
# backfill and hourly pipelines write these rows as-is, and the feature
# store fills them in naturally as later runs re-observe the same hours with
# their outcome now known. Training drops per-horizon NaNs itself, so a row
# missing only e.g. the 72h target still contributes to 24h/48h training).
# --------------------------------------------------------------------------
def build_training_table(city: str, raw: pd.DataFrame,
                          horizons=FORECAST_HORIZONS_H) -> pd.DataFrame:
    feats = engineer_features(raw)
    table = build_targets(feats, horizons)
    table = table.dropna(subset=["us_aqi_lag24"])   # drop only the warm-up edge
    table.insert(0, "city", city)
    return table
