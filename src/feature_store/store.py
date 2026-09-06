"""
Feature store: Hopsworks-backed, with a local-parquet fallback.

If HOPSWORKS_API_KEY / HOPSWORKS_PROJECT are set (see config.py /
.env.example), every call talks to a real Hopsworks feature group, one per
city, keyed on `time`. Otherwise it transparently falls back to a per-city
parquet file under data/, so pipelines and the dashboard work identically
either way - only config.py changes when Hopsworks is wired up for real.

All functions take/return a DataFrame indexed by a DatetimeIndex named
"time" (what src.data.fetch / src.features.engineering already produce).
"""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd

from config import (
    DATA_DIR, HOPSWORKS_ENABLED, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
    FEATURE_GROUP_VERSION, feature_group_name,
)

# The Hopsworks Query Service (Arrow Flight) can throw transient
# "Set changed size during iteration" server errors for a couple of minutes
# right after a large materialization job finishes, while its internal view
# of the table's file set is still converging. Backs off up to ~4 minutes
# total, which has been enough in practice for a 365-day/5-city table.
_READ_RETRY_DELAYS_S = [10, 20, 30, 45, 60, 90]

# Occasionally the query service doesn't error at all - it just never sends a
# data message, and pyarrow's do_get() has no client-side timeout, so it can
# hang indefinitely. Bound each attempt in a worker thread so a stuck call
# can still be treated as a failure and retried, instead of freezing the
# whole pipeline/dashboard. The stuck thread itself is abandoned (pyarrow
# Flight calls aren't cleanly cancellable) but that's a one-off leak, not a
# repeated one, since retries reuse the shared, already-open connection.
_READ_TIMEOUT_S = 150
_read_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hopsworks-read")

# Writes can also hit transient connection errors talking to the Hopsworks
# REST API (seen in practice: a ConnectionError/RemoteDisconnected while
# launching the materialization job). Retrying is safe because inserts are
# Hudi upserts keyed on `time` - resubmitting the same rows just overwrites
# them with identical values, never duplicates. Shorter backoff than reads
# since this failure mode has shown up as a fast connection drop, not a slow
# server-side convergence issue.
_WRITE_RETRY_DELAYS_S = [15, 30, 60]

_project = None


def get_hopsworks_project():
    """Shared Hopsworks login, reused by the model registry too."""
    global _project
    if _project is None:
        import hopsworks
        _project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    return _project


def _get_or_create_feature_group(city: str, primary_key=("time",)):
    fs = get_hopsworks_project().get_feature_store()
    return fs.get_or_create_feature_group(
        name=feature_group_name(city),
        version=FEATURE_GROUP_VERSION,
        description=f"Hourly AQI + weather features/targets for {city}",
        primary_key=list(primary_key),
        event_time="time",
        time_travel_format="HUDI",  # avoids requiring the `delta` lib client-side
        statistics_config={"enabled": False, "histograms": False, "correlations": False},
    )


def _local_path(city: str):
    return DATA_DIR / f"{city}_features.parquet"


def write_features(city: str, df: pd.DataFrame) -> None:
    """Upsert rows into the city's feature group (or local parquet fallback)."""
    if df.empty:
        return
    flat = df.reset_index() if df.index.name == "time" else df.copy()
    flat["time"] = pd.to_datetime(flat["time"])

    if HOPSWORKS_ENABLED:
        fg = _get_or_create_feature_group(city)
        for i, delay in enumerate([0, *_WRITE_RETRY_DELAYS_S]):
            if delay:
                time.sleep(delay)
            try:
                fg.insert(flat, write_options={"wait_for_job": True})
                return
            except Exception:
                if i == len(_WRITE_RETRY_DELAYS_S):
                    raise

    path = _local_path(city)
    if path.exists():
        existing = pd.read_parquet(path)
        if "time" not in existing.columns:      # tolerate a time-indexed file
            existing = existing.reset_index().rename(columns={existing.index.name or "index": "time"})
        flat = pd.concat([existing, flat], ignore_index=True)
    flat = flat.drop_duplicates(subset="time", keep="last").sort_values("time")
    flat.to_parquet(path, index=False)


def read_features(city: str) -> pd.DataFrame:
    """Read the full feature history for a city, indexed by time."""
    if HOPSWORKS_ENABLED:
        fg = _get_or_create_feature_group(city)
        for i, delay in enumerate([0, *_READ_RETRY_DELAYS_S]):
            if delay:
                time.sleep(delay)
            try:
                df = _read_pool.submit(fg.read).result(timeout=_READ_TIMEOUT_S)
                break
            except (Exception, FutureTimeoutError):
                if i == len(_READ_RETRY_DELAYS_S):
                    raise
    else:
        path = _local_path(city)
        if not path.exists():
            raise FileNotFoundError(
                f"No local features for {city!r} yet - run the backfill pipeline first."
            )
        df = pd.read_parquet(path)

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        # Hopsworks returns tz-aware (UTC) timestamps; everything else in this
        # project (Open-Meteo fetches, the local-parquet fallback) is naive
        # local time. Normalize to naive here so callers never have to know
        # which backend served the data - see future_weather_at, which
        # compares this index against a freshly-fetched (naive) forecast.
        df["time"] = df["time"].dt.tz_localize(None)
    return df.set_index("time").sort_index()
