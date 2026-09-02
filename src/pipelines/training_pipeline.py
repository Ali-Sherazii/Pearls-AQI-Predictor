"""
Daily training pipeline: read features from the store, train + evaluate the
per-horizon model for every configured city, and save it to the model
registry - even if it loses to the persistence baseline, so that's visible
rather than hidden.

Run:  python -m src.pipelines.training_pipeline
"""

from __future__ import annotations
import argparse

from config import CITIES, FORECAST_HORIZONS_H
from src.feature_store.store import read_features
from src.models.train import train_and_evaluate
from src.models.registry import save_model


def run_city(city: str, horizons=FORECAST_HORIZONS_H):
    df = read_features(city)
    for h in horizons:
        bundle, metrics, baseline = train_and_evaluate(df, h)
        path = save_model(city, h, bundle, metrics)
        flag = "beats" if metrics["rmse"] < baseline["rmse"] else "LOSES TO"
        print(f"{city} +{h}h: rmse={metrics['rmse']:.2f} (baseline {baseline['rmse']:.2f}) "
              f"mae={metrics['mae']:.2f} r2={metrics['r2']:.3f}  "
              f"[{flag} persistence]  -> {path}")


def main(cities: list[str] | None = None, horizons=FORECAST_HORIZONS_H):
    for city in (cities or list(CITIES)):
        run_city(city, horizons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", dest="cities",
                         help="restrict to one city (repeatable); default = all configured cities")
    args = parser.parse_args()
    main(args.cities)
