"""
Honest evaluation with time-series cross-validation.

A single train/test split can be misleading when you only have one year of
data (you end up testing on one season). TimeSeriesSplit tests on several
later slices in turn, so the score averages over winter AND summer.

For each horizon it reports mean +/- std across folds for the persistence
baseline, Ridge and Random Forest.

Run:  python train_cv.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA = Path("data/lahore_features.parquet")   # change city name if needed
N_SPLITS = 5


def metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))


def main():
    df = pd.read_parquet(DATA).sort_index().dropna()
    target_cols = [c for c in df.columns if c.startswith("target_aqi_t")]
    feature_cols = [c for c in df.columns
                    if c not in target_cols and c != "city"
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"{len(df)} rows | {len(feature_cols)} features | {N_SPLITS} time folds\n")

    def make_models():
        return {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "random_forest": RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=42),
        }

    rows = []
    for target in target_cols:
        y = df[target]
        # collect per-fold metrics
        acc = {"persistence": [], "ridge": [], "random_forest": []}
        for tr, te in tss.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            acc["persistence"].append(metrics(yte, Xte["us_aqi"].to_numpy()))
            for name, model in make_models().items():
                model.fit(Xtr, ytr)
                acc[name].append(metrics(yte, model.predict(Xte)))
        for name, vals in acc.items():
            arr = np.array(vals)  # folds x (rmse,mae,r2)
            mean, std = arr.mean(0), arr.std(0)
            rows.append((target, name,
                         mean[0], std[0], mean[1], mean[2], std[2]))

    res = pd.DataFrame(rows, columns=[
        "horizon", "model", "rmse", "rmse_std", "mae", "r2", "r2_std"])
    pd.set_option("display.float_format", lambda v: f"{v:7.2f}")
    print(res.to_string(index=False), "\n")

    print("Best per horizon (mean RMSE across folds):")
    for target in target_cols:
        sub = res[res.horizon == target]
        base = sub[sub.model == "persistence"]["rmse"].iloc[0]
        best = sub.loc[sub["rmse"].idxmin()]
        verdict = "BEATS baseline" if best["rmse"] < base else "loses to baseline"
        print(f"  {target:>16}: {best['model']:<14} "
              f"rmse={best['rmse']:6.2f}+/-{best['rmse_std']:4.1f} "
              f"r2={best['r2']:6.3f}  (base {base:6.2f})  [{verdict}]")


if __name__ == "__main__":
    main()