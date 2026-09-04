"""
Hourly feature pipeline: fetch recent observations, engineer features, upsert
into the feature store for every configured city.

A 10-day rolling window gives the lag24/roll24 features enough context, and
guarantees rows written ~72h+ ago now have their targets filled in too, since
build_training_table only drops the warm-up edge, not rows with unresolved
targets (see src/features/engineering.py).

Run:  python -m src.pipelines.feature_pipeline
"""

from __future__ import annotations
import argparse
import sys

from config import CITIES
from src.data.fetch import fetch_recent_raw
from src.features.engineering import build_training_table
from src.feature_store.store import write_features

RECENT_WINDOW_DAYS = 10


def run_city(city: str, window_days: int = RECENT_WINDOW_DAYS):
    raw = fetch_recent_raw(city, past_days=window_days)
    table = build_training_table(city, raw)
    write_features(city, table)
    print(f"{city}: upserted {len(table)} rows through {table.index.max()}")


def main(cities: list[str] | None = None):
    # One city's failure (a still-flaky upstream API, an intermittent
    # Hopsworks error) shouldn't cost the other four their hourly update -
    # run each independently and only fail the job at the end if any did.
    failed = []
    for city in (cities or list(CITIES)):
        try:
            run_city(city)
        except Exception as exc:
            print(f"{city}: FAILED - {exc!r}")
            failed.append(city)
    if failed:
        print(f"\n{len(failed)} of {len(cities or CITIES)} cities failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", dest="cities",
                         help="restrict to one city (repeatable); default = all configured cities")
    args = parser.parse_args()
    main(args.cities)
