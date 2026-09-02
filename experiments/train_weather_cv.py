"""
The real experiment: does FUTURE weather let us beat persistence?

For each horizon h, we add the weather AT the target hour (t+h) plus the
cumulative rain and average wind between now and then. These are derived by
shifting the weather columns already in the parquet, so no re-fetch is needed.
Models predict the CHANGE from now (delta framing) and are scored against the
absolute future AQI, head-to-head with persistence, via time-series CV.

NOTE: shifting the archive gives ACTUAL future weather (a perfect forecast), so
these numbers are an optimistic ceiling. Real deployment uses forecast weather.

Run:  python train_weather_cv.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA = Path("data/lahore_features.parquet")
N_SPLITS = 5
WX = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
      "wind_direction_10m", "surface_pressure", "precipitation"]


def metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))


def make_models():
    return {
        "ridge_wx": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "rf_wx": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    }


def build_future_weather(df, h):
    """Weather at the target hour t+h, plus cumulative rain / mean wind to then."""
    fut = pd.DataFrame(index=df.index)
    for w in WX:
        if w in df.columns:
            fut[f"{w}_t{h}h"] = df[w].shift(-h)            # weather AT target hour
    if "precipitation" in df.columns:
        fut[f"precip_sum_next{h}h"] = df["precipitation"].rolling(h).sum().shift(-h)
    if "wind_speed_10m" in df.columns:
        fut[f"wind_mean_next{h}h"] = df["wind_speed_10m"].rolling(h).mean().shift(-h)
    return fut


def main():
    df = pd.read_parquet(DATA).sort_index()
    target_cols = [c for c in df.columns if c.startswith("target_aqi_t")]
    base_feats = [c for c in df.columns
                  if c not in target_cols and c != "city"
                  and pd.api.types.is_numeric_dtype(df[c])]
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"{len(df)} rows | {len(base_feats)} present features + future weather | "
          f"{N_SPLITS} folds\n")

    rows = []
    for target in target_cols:
        h = int(target.replace("target_aqi_t", "").replace("h", ""))
        fut = build_future_weather(df, h)
        d = pd.concat([df[base_feats + [target]], fut], axis=1).dropna()

        feat_cols = base_feats + list(fut.columns)
        X = d[feat_cols]
        now = d["us_aqi"]
        y_delta = d[target] - now
        y_abs = d[target]

        acc = {"persistence": [], "ridge_wx": [], "rf_wx": []}
        for tr, te in tss.split(X):
            now_te = now.iloc[te].to_numpy()
            yte = y_abs.iloc[te].to_numpy()
            acc["persistence"].append(metrics(yte, now_te))
            for name, model in make_models().items():
                model.fit(X.iloc[tr], y_delta.iloc[tr])
                acc[name].append(metrics(yte, now_te + model.predict(X.iloc[te])))

        for name, vals in acc.items():
            arr = np.array(vals)
            m, s = arr.mean(0), arr.std(0)
            rows.append((target, name, m[0], s[0], m[1], m[2], s[2]))

    res = pd.DataFrame(rows, columns=[
        "horizon", "model", "rmse", "rmse_std", "mae", "r2", "r2_std"])
    pd.set_option("display.float_format", lambda v: f"{v:7.2f}")
    print(res.to_string(index=False), "\n")

    print("Best per horizon (mean RMSE across folds):")
    for target in target_cols:
        sub = res[res.horizon == target]
        base = sub[sub.model == "persistence"]["rmse"].iloc[0]
        best = sub.loc[sub["rmse"].idxmin()]
        gain = (base - best["rmse"]) / base * 100
        verdict = f"BEATS baseline by {gain:.0f}%" if best["rmse"] < base else "loses to baseline"
        print(f"  {target:>16}: {best['model']:<10} "
              f"rmse={best['rmse']:6.2f}+/-{best['rmse_std']:4.1f} "
              f"r2={best['r2']:6.3f}  (base {base:6.2f})  [{verdict}]")


if __name__ == "__main__":
    main()
