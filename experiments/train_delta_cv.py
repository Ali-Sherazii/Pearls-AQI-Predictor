"""
Same honest cross-validated evaluation, but the models predict the CHANGE in
AQI from now (delta = future_AQI - current_AQI) instead of the absolute value.
The final prediction is reconstructed as current_AQI + predicted_delta, so it's
anchored on persistence by construction and only has to learn the deviation.

Everything is still scored against the ABSOLUTE future AQI, head-to-head with
the persistence baseline, so the comparison is apples-to-apples.

Run:  python train_delta_cv.py
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


def metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))


def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=42),
    }


def main():
    df = pd.read_parquet(DATA).sort_index().dropna()
    target_cols = [c for c in df.columns if c.startswith("target_aqi_t")]
    feature_cols = [c for c in df.columns
                    if c not in target_cols and c != "city"
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    now = df["us_aqi"]
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"{len(df)} rows | {len(feature_cols)} features | "
          f"{N_SPLITS} folds | DELTA framing\n")

    rows = []
    for target in target_cols:
        y_abs = df[target]
        y_delta = y_abs - now              # what the models actually learn
        acc = {"persistence": [], "ridge_delta": [], "rf_delta": []}
        for tr, te in tss.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            now_te = now.iloc[te].to_numpy()
            yte_abs = y_abs.iloc[te].to_numpy()

            acc["persistence"].append(metrics(yte_abs, now_te))
            for label, model in zip(("ridge_delta", "rf_delta"),
                                    make_models().values()):
                model.fit(Xtr, y_delta.iloc[tr])
                pred_abs = now_te + model.predict(Xte)   # reconstruct
                acc[label].append(metrics(yte_abs, pred_abs))

        for name, vals in acc.items():
            arr = np.array(vals)
            mean, std = arr.mean(0), arr.std(0)
            rows.append((target, name, mean[0], std[0], mean[1], mean[2], std[2]))

    res = pd.DataFrame(rows, columns=[
        "horizon", "model", "rmse", "rmse_std", "mae", "r2", "r2_std"])
    pd.set_option("display.float_format", lambda v: f"{v:7.2f}")
    print(res.to_string(index=False), "\n")

    print("Best per horizon (mean RMSE across folds):")
    for target in target_cols:
        sub = res[res.horizon == target]
        base = sub[sub.model == "persistence"]["rmse"].iloc[0]
        best = sub.loc[sub["rmse"].idxmin()]
        verdict = "BEATS baseline" if best["rmse"] < base - 1e-9 else "loses to baseline"
        print(f"  {target:>16}: {best['model']:<14} "
              f"rmse={best['rmse']:6.2f}+/-{best['rmse_std']:4.1f} "
              f"r2={best['r2']:6.3f}  (base {base:6.2f})  [{verdict}]")


if __name__ == "__main__":
    main()
