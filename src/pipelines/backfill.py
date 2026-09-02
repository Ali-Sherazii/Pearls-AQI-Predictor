"""
Backfill historical (features, targets) into the feature store.

Run for a range of past days per city so there's enough history to train on.
Safe to re-run: writes upsert by `time`, so re-backfilling just refreshes rows.

Run:  python -m src.pipelines.backfill [--city lahore] [--history-days 365]
"""

from __future__ import annotations
import argparse

from config import CITIES, HISTORY_DAYS
from src.data.fetch import fetch_raw_history
from src.features.engineering import build_training_table
from src.feature_store.store import write_features


def backfill_city(city: str, history_days: int = HISTORY_DAYS):
    print(f"Backfilling {history_days} days for {city} ...")
    raw = fetch_raw_history(city, history_days)
    table = build_training_table(city, raw)
    write_features(city, table)
    print(f"  rows={len(table)} cols={table.shape[1]} "
          f"range={table.index.min()} -> {table.index.max()}")
    return table


def main(cities: list[str] | None = None, history_days: int = HISTORY_DAYS):
    for city in (cities or list(CITIES)):
        backfill_city(city, history_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", dest="cities",
                         help="restrict to one city (repeatable); default = all configured cities")
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args()
    main(args.cities, args.history_days)
