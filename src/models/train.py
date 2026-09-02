"""
Training + honest evaluation for one city's feature table.

Time-ordered train (80%) / test (20%) split, never shuffled. On the train
slice only:

  1. Model selection: time-series CV produces OUT-OF-FOLD delta predictions
     for several Random Forest / HistGradientBoosting configs (the recipe
     experiments/train_weather_cv.py validated as the winner, generalized to
     search a few configs instead of one fixed one). Each is reconstructed to
     absolute AQI and scored against actuals to pick the best family.
  2. Shrinkage calibration: using that SAME pool of out-of-fold predictions
     (never touching test), grid-search a scalar alpha so the deployed
     prediction is `current + alpha * predicted_delta`. alpha=0 reproduces
     persistence exactly and is always in the search grid, so calibration
     can never choose something that scores worse than the baseline on this
     pool - this is what fixes cities/horizons where the raw model overshoots.

Reusing the same out-of-fold pool for both steps (rather than carving out a
dedicated validation slice) matters: an earlier version held out a separate
20% validation chunk and it measurably hurt test RMSE by shrinking the
80% train slice down to 60% - on ~8-9k rows/city that's real signal lost for
comparatively little calibration benefit.

Finally: fit the winning config on the full 80% train, evaluate on the
untouched 20% test with the calibrated alpha for the metrics that get
reported, then refit on ALL data (same alpha) for the model that actually
gets deployed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features.engineering import build_future_weather

TEST_FRAC = 0.2
CV_SPLITS = 4
RANDOM_STATE = 42
ALPHA_GRID = np.round(np.arange(0.0, 1.25, 0.05), 2)

# A few configs per family rather than one fixed recipe - picked per
# city/horizon by CV instead of assumed to transfer from Lahore (where the
# original recipe was validated) to every city.
CANDIDATES = {
    "rf_deep": lambda: RandomForestRegressor(
        n_estimators=150, max_depth=14, n_jobs=-1, random_state=RANDOM_STATE),
    "rf_reg": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=3,
        n_jobs=-1, random_state=RANDOM_STATE),
    "hgb_a": lambda: HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE),
    "hgb_b": lambda: HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.03,
        l2_regularization=1.0, random_state=RANDOM_STATE),
}


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


def _oof_predictions(X: pd.DataFrame, y_delta: pd.Series) -> dict[str, np.ndarray]:
    """Out-of-fold delta predictions for every candidate over the train
    slice. Rows in the first fold's training-only prefix never get a
    held-out prediction and stay NaN."""
    tss = TimeSeriesSplit(n_splits=CV_SPLITS)
    oof = {name: np.full(len(X), np.nan) for name in CANDIDATES}
    for tr, va in tss.split(X):
        for name, make in CANDIDATES.items():
            model = make()
            model.fit(X.iloc[tr], y_delta.iloc[tr])
            oof[name][va] = model.predict(X.iloc[va])
    return oof


def _best_alpha(now: np.ndarray, y_abs: np.ndarray,
                 delta_pred: np.ndarray) -> tuple[float, float]:
    """Grid-search a persistence-shrinkage factor: final = current +
    alpha * predicted_delta. alpha=0 reproduces persistence exactly, so the
    chosen alpha is guaranteed to score at least as well as the baseline on
    whatever pool of (now, y_abs, delta_pred) it's calibrated against."""
    best_alpha, best_rmse = 1.0, np.inf
    for alpha in ALPHA_GRID:
        rmse = np.sqrt(mean_squared_error(y_abs, now + alpha * delta_pred))
        if rmse < best_rmse:
            best_alpha, best_rmse = float(alpha), rmse
    return best_alpha, best_rmse


def train_and_evaluate(df: pd.DataFrame, h: int, test_frac: float = TEST_FRAC):
    """Returns (bundle, metrics, baseline). `bundle` is what registry.save_model
    persists; `metrics`/`baseline` are the honest held-out numbers (on the
    untouched test slice) to record alongside it."""
    d, feat_cols, target = _prepare(df, h)
    X, y_abs, now = d[feat_cols], d[target], d["us_aqi"]
    y_delta = y_abs - now

    n = len(d)
    cut = int(n * (1 - test_frac))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr_delta = y_delta.iloc[:cut]
    now_tr_arr, now_te = now.iloc[:cut].to_numpy(), now.iloc[cut:]
    ytr_abs_arr, yte_abs = y_abs.iloc[:cut].to_numpy(), y_abs.iloc[cut:]

    baseline = evaluate(yte_abs, now_te.to_numpy())

    oof = _oof_predictions(Xtr, ytr_delta)
    valid = ~np.isnan(next(iter(oof.values())))
    now_oof, yabs_oof = now_tr_arr[valid], ytr_abs_arr[valid]

    cv_rmse = {name: np.sqrt(mean_squared_error(yabs_oof, now_oof + oof[name][valid]))
               for name in CANDIDATES}
    best_name = min(cv_rmse, key=cv_rmse.get)
    alpha, _ = _best_alpha(now_oof, yabs_oof, oof[best_name][valid])

    train_model = CANDIDATES[best_name]()
    train_model.fit(Xtr, ytr_delta)
    pred_abs = now_te.to_numpy() + alpha * train_model.predict(Xte)
    metrics = evaluate(yte_abs, pred_abs)

    # refit the winning recipe on ALL data for the model that actually gets
    # deployed; alpha travels with it in the bundle so serving code applies
    # the same shrinkage to the live prediction.
    final_model = CANDIDATES[best_name]()
    final_model.fit(X, y_delta)

    bundle = {
        "model": final_model,
        "feature_cols": feat_cols,
        "horizon_h": h,
        "framing": "delta",
        "alpha": alpha,
        "model_family": best_name,
    }
    return bundle, metrics, baseline
