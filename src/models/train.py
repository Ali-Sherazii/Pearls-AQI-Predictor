"""
Training + honest evaluation for one city's feature table.

For each horizon: build the delta-framing + future-weather dataset that
experiments/train_weather_cv.py validated as the winner, hold out the most
recent TEST_FRAC of rows to score against the persistence baseline, then
refit on all available data for the model that actually gets deployed.

n_estimators/max_depth are capped relative to the original experiment
(300 trees, unbounded depth -> ~180MB per model) - see report/report.md for
the RMSE-vs-size tradeoff that justified this.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features.engineering import build_future_weather

TEST_FRAC = 0.2
N_ESTIMATORS = 150
MAX_DEPTH = 14
RANDOM_STATE = 42


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _prepare(df: pd.DataFrame, h: int):
    target = f"target_aqi_t{h}h"
    base_feats = [c for c in df.columns
                  if c != "city" and not c.startswith("target_aqi_t")
                  and pd.api.types.is_numeric_dtype(df[c])]
    fut = build_future_weather(df, h)
    d = pd.concat([df[base_feats + [target]], fut], axis=1).dropna()
    feat_cols = base_feats + list(fut.columns)
    return d, feat_cols, target


def _make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        n_jobs=-1, random_state=RANDOM_STATE,
    )


def train_and_evaluate(df: pd.DataFrame, h: int, test_frac: float = TEST_FRAC):
    """Returns (bundle, metrics, baseline). `bundle` is what registry.save_model
    persists; `metrics`/`baseline` are the honest held-out numbers to record
    alongside it."""
    d, feat_cols, target = _prepare(df, h)

    cut = int(len(d) * (1 - test_frac))
    train, test = d.iloc[:cut], d.iloc[cut:]
    now_tr, now_te = train["us_aqi"], test["us_aqi"]
    yte_abs = test[target]

    baseline = evaluate(yte_abs, now_te.to_numpy())

    holdout_model = _make_model()
    holdout_model.fit(train[feat_cols], train[target] - now_tr)
    pred_abs = now_te.to_numpy() + holdout_model.predict(test[feat_cols])
    metrics = evaluate(yte_abs, pred_abs)

    # refit on ALL data for the model that actually gets deployed
    final_model = _make_model()
    final_model.fit(d[feat_cols], d[target] - d["us_aqi"])

    bundle = {
        "model": final_model,
        "feature_cols": feat_cols,
        "horizon_h": h,
        "framing": "delta",
    }
    return bundle, metrics, baseline
