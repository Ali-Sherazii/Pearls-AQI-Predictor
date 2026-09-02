"""
Deep learning candidate: does a small MLP beat Ridge/RF-delta+weather?

Same time-series CV harness and delta+future-weather framing as
train_weather_cv.py, with a Keras MLP added as a 4th candidate. This is
deliberately a separate, standalone experiment (not part of the production
src/models/train.py) - TensorFlow is a heavy dependency the deployed
dashboard/pipelines don't need, and this script exists purely to give the
report an honest statistical-to-deep-learning comparison, per the brief's
guideline to try a variety of forecasting models.

Requires: pip install tensorflow-cpu (not in requirements.txt - see above)

Run:  python experiments/train_nn_cv.py
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

import tensorflow as tf
from tensorflow import keras

DATA = Path("data/lahore_features.parquet")
N_SPLITS = 5
WX = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
      "wind_direction_10m", "surface_pressure", "precipitation"]

tf.random.set_seed(42)


def metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))


def build_future_weather(df, h):
    fut = pd.DataFrame(index=df.index)
    for w in WX:
        if w in df.columns:
            fut[f"{w}_t{h}h"] = df[w].shift(-h)
    if "precipitation" in df.columns:
        fut[f"precip_sum_next{h}h"] = df["precipitation"].rolling(h).sum().shift(-h)
    if "wind_speed_10m" in df.columns:
        fut[f"wind_mean_next{h}h"] = df["wind_speed_10m"].rolling(h).mean().shift(-h)
    return fut


def make_mlp(n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    df = pd.read_parquet(DATA).sort_index()
    target_cols = [c for c in df.columns if c.startswith("target_aqi_t")]
    base_feats = [c for c in df.columns
                  if c not in target_cols and c != "city"
                  and pd.api.types.is_numeric_dtype(df[c])]
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"{len(df)} rows | {len(base_feats)} present features + future weather | "
          f"{N_SPLITS} folds | statistical vs deep learning\n")

    rows = []
    for target in target_cols:
        h = int(target.replace("target_aqi_t", "").replace("h", ""))
        fut = build_future_weather(df, h)
        d = pd.concat([df[base_feats + [target]], fut], axis=1).dropna()

        feat_cols = base_feats + list(fut.columns)
        X = d[feat_cols].to_numpy()
        now = d["us_aqi"].to_numpy()
        y_delta = (d[target] - d["us_aqi"]).to_numpy()
        y_abs = d[target].to_numpy()

        acc = {"persistence": [], "ridge_wx": [], "rf_wx": [], "mlp_wx": []}
        for tr, te in tss.split(X):
            now_te, yte = now[te], y_abs[te]
            acc["persistence"].append(metrics(yte, now_te))

            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            ridge.fit(X[tr], y_delta[tr])
            acc["ridge_wx"].append(metrics(yte, now_te + ridge.predict(X[te])))

            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
            rf.fit(X[tr], y_delta[tr])
            acc["rf_wx"].append(metrics(yte, now_te + rf.predict(X[te])))

            scaler = StandardScaler().fit(X[tr])
            mlp = make_mlp(X.shape[1])
            mlp.fit(scaler.transform(X[tr]), y_delta[tr],
                    epochs=30, batch_size=64, verbose=0,
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
            pred = mlp.predict(scaler.transform(X[te]), verbose=0).ravel()
            acc["mlp_wx"].append(metrics(yte, now_te + pred))

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
        verdict = "BEATS baseline" if best["rmse"] < base else "loses to baseline"
        print(f"  {target:>16}: {best['model']:<10} "
              f"rmse={best['rmse']:6.2f}+/-{best['rmse_std']:4.1f} "
              f"r2={best['r2']:6.3f}  (base {base:6.2f})  [{verdict}]")


if __name__ == "__main__":
    main()
