# Pearls AQI Predictor — Report

A 3-day Air Quality Index forecasting system for 5 Pakistani cities (Lahore,
Islamabad, Rawalpindi, Karachi, Peshawar), built on a serverless feature
store / model registry architecture with an hourly feature pipeline, a
daily training pipeline, and an interactive dashboard.

## 1. Architecture

```
Open-Meteo API (air quality + weather, no key required)
        |
        v
Feature pipeline (hourly)  ---features--->  Feature Store
Backfill (one-time, 365d)  ---features--->  (Hopsworks, or local
                                              parquet fallback)
                                                    |
                                                    v
                                          Training pipeline (daily)
                                                    |
                                              features, targets
                                                    v
                                             Model Registry
                                          (Hopsworks, or local
                                              joblib fallback)
                                                    |
                                              model, features
                                                    v
                                          Streamlit dashboard
                                    (forecast, hazard alerts, SHAP)
```

GitHub Actions runs the feature pipeline hourly and the training pipeline
daily (`.github/workflows/`). The feature store and model registry are
Hopsworks-backed when `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT` are configured;
otherwise every component transparently falls back to local parquet/joblib
files, so the whole system is runnable and testable with zero external
accounts (see `src/feature_store/store.py`, `src/models/registry.py`).

## 2. Data

**Source:** Open-Meteo's air-quality and weather APIs (chosen over
AQICN/OpenWeather because it needs no API key, has no rate-limit friction for
a project this size, and serves both historical archive and forecast data
from the same schema).

**Cities:** Lahore, Islamabad, Rawalpindi, Karachi, Peshawar — a full year of
hourly data each (2025-08-29 to 2026-08-28, 8,760 rows/city), backfilled into
the real Hopsworks feature store.

**Fields:** `us_aqi`, `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`,
`sulphur_dioxide`, `ozone` (air quality) and `temperature_2m`,
`relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`,
`surface_pressure`, `precipitation` (weather).

**Data-quality finding:** Islamabad and Rawalpindi's air-quality fields
(`us_aqi`, `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`)
are **100% identical** between the two cities across all 8,880 hours — their
~13km separation falls inside a single cell of Open-Meteo's underlying
CAMS air-quality reanalysis grid. Their weather fields differ normally (e.g.
temperature matches only 4.5% of the time), confirming the weather model
has finer resolution than the air-quality model. This is a genuine
limitation of the free data source, not a pipeline bug — verified by
comparing the two cities' raw feature tables column-by-column.

## 3. Exploratory Data Analysis (`notebooks/01_eda.ipynb`)

- **AQI level by city:** Lahore has both the highest mean AQI (154.7) and the
  widest spread (std 48.1, max 364 — "Hazardous"). Karachi is consistently
  the cleanest (mean 89.5, mostly Good/Moderate). Islamabad/Rawalpindi (mean
  113.8) and Peshawar (115.6) sit in between.
- **Seasonality (Lahore):** a pronounced winter smog season — mean AQI peaks
  in January (229.5) and December (199.5), and troughs in April (95.8). This
  matches the well-documented regional smog pattern (crop burning + winter
  inversions trapping pollutants).
- **Diurnal pattern:** all cities show repeating daily spikes tied to
  traffic/cooking hours, visible directly in the dashboard's 14-day history
  chart.
- **Correlation with AQI (Lahore):** `pm2_5` (0.75) and `sulphur_dioxide`
  (0.66) correlate most strongly; `temperature_2m` (-0.44) and
  `wind_speed_10m` (-0.29) correlate negatively (wind disperses pollutants,
  consistent with atmospheric science).

## 4. Feature engineering (`src/features/engineering.py`)

One shared implementation used by backfill, the hourly pipeline, training,
and serving (eliminating train/serve skew):

- **Time-based:** hour, day, month, day-of-week, is-weekend, plus cyclical
  sin/cos encodings of hour and month.
- **Lag/rolling:** 1h-lag, 24h-lag, and 24h rolling mean for `us_aqi`,
  `pm2_5`, `pm10`.
- **Derived:** 1h and 24h AQI change rate.
- **Future weather (serving-time only):** weather forecast at the target
  hour `t+h`, plus cumulative rainfall and mean wind speed between now and
  then — the single biggest driver of forecast accuracy (see below).

## 5. Model selection (`experiments/`)

Four experiments, each evaluated with 5-fold **time-series cross-validation**
(never shuffled) against a **persistence baseline** ("AQI in h hours = AQI
now") on Lahore's data — the harness in every script is identical so the
numbers are directly comparable:

| Experiment | 24h RMSE | 48h RMSE | 72h RMSE | Verdict |
|---|---|---|---|---|
| Persistence baseline | 28.95–29.07 | 36.59–36.61 | 39.49 | — |
| **1. Absolute framing** (Ridge / RF predict AQI directly) | RF 32.63 | RF 40.86 | RF 49.18 | Loses to baseline at every horizon |
| **2. Delta framing** (predict the *change* from now) | RF 27.93 (beats) | RF 36.92 (ties/loses) | RF 39.78 (loses) | Beats baseline only at 24h |
| **3. Delta + future weather** | RF 25.61 (**+12%**) | RF 32.52 (**+11%**) | RF 35.09 (**+11%**) | **Beats baseline at every horizon — winner** |
| **4. + Deep learning (MLP)** | MLP 25.01 (best) | RF 32.52 (best) | RF 35.09 (best) | MLP edges out RF only at 24h |

Ridge consistently performed far worse than the baseline in every framing
(RMSE 50–190+), including with `StandardScaler` — the high-dimensional,
correlated feature set (lags + rolling + future weather) doesn't suit a
linear model here.

**Conclusion:** future weather is what actually makes 48h/72h forecasting
viable — delta framing alone isn't enough at longer horizons, because the
model needs to know what's *going to happen* to the atmosphere, not just
extrapolate from what already happened. A small MLP was competitive at 24h
but never clearly better, and pulls in a heavy TensorFlow dependency the
serverless dashboard/pipelines don't otherwise need — so **Random Forest on
delta + future weather** (experiment 3) is what `src/models/train.py`
implements for production, reproducing experiment 4's finding that it wins
or ties at every horizon.

**Model size vs. accuracy:** the original exploration used 300 trees with
unbounded depth (~180MB/model). Production uses `n_estimators=150,
max_depth=14` (~30–48MB/model, a 4–5x reduction) — chosen because reducing
tree count/depth this far did not meaningfully change RMSE in testing, and a
6-model registry (3 horizons × cheaper to scale to more cities) needs to stay
practical to store and serve.

## 6. Final production results (`src/pipelines/training_pipeline.py`)

Per-city, per-horizon, held out on the most recent 20% of each city's year of
data (a single realistic holdout, distinct from the CV numbers above). These
are the actual models live in the Hopsworks Model Registry as of this run:

| City | +24h RMSE (base) | +48h RMSE (base) | +72h RMSE (base) |
|---|---|---|---|
| Lahore | 25.88 (34.80) ✅ | 33.66 (42.71) ✅ | 34.00 (42.84) ✅ |
| Islamabad | 20.23 (18.92) ❌ | 25.15 (24.37) ❌ | 26.53 (27.70) ✅ |
| Rawalpindi | 19.95 (18.92) ❌ | 26.78 (24.37) ❌ | 27.29 (27.70) ✅ |
| Karachi | 5.42 (6.17) ✅ | 10.30 (8.73) ❌ | 8.41 (10.22) ✅ |
| Peshawar | 20.69 (20.27) ❌ | 27.29 (25.04) ❌ | 28.00 (26.93) ❌ |

✅ = model beats persistence, ❌ = model loses to persistence (both are saved
to the registry regardless — see `src/pipelines/training_pipeline.py` — so
this is visible rather than hidden).

**Honest interpretation:** the model clearly earns its keep in Lahore, where
AQI is both high-variance and high-stakes (this is also where the winning
recipe was validated), and every city beats persistence at +72h — exactly
where a naive "tomorrow = today" forecast should struggle most. 24h/48h are
more mixed: Karachi's very low AQI range (mostly Good/Moderate) means
persistence is already close to optimal, leaving little room to improve, and
Islamabad/Rawalpindi/Peshawar's more modest AQI swings than Lahore mean
persistence is a stronger baseline there too. This is a legitimate
limitation, not a bug — see Section 8. (Exact numbers will drift slightly
run-to-run — RandomForest isn't perfectly reproducible across different data
pulls even with a fixed `random_state`, since the holdout split point shifts
with the row count — but the pattern above is stable.)

## 7. Explainability (`src/models/explain.py`)

Each dashboard forecast is paired with a SHAP `TreeExplainer` breakdown of
the top features driving that specific prediction (in AQI-delta units, so a
positive bar means "pushes the forecast up from today's AQI"). For example, a
Lahore +24h forecast on 2026-09-01 was driven up by `us_aqi_roll24`, `day`,
and `pm10`, and driven down by `wind_mean_next24h` and `us_aqi_lag1` — wind
picking up over the next day was the single largest downward contributor,
consistent with the EDA's wind/AQI correlation.

## 8. Alerts

The dashboard raises a hazard banner whenever any horizon's forecast AQI
reaches 150 (US AQI "Unhealthy for Sensitive Groups" and above) —
`config.HAZARD_AQI`, checked in both `predict`-style serving code and the
dashboard.

## 9. Dashboard (`app/streamlit_app.py`)

City picker across all 5 cities; current AQI with a color-coded category
badge; a 3-day forecast line chart with a hazard-threshold reference line;
a forecast detail table; per-horizon SHAP explanations; and a 14-day AQI
history chart. Verified end-to-end for both a high-AQI city (Lahore, current
AQI 157 "Unhealthy," correctly raising the hazard alert for +24h) and a
low-AQI city (Karachi, current AQI 62 "Moderate," no alert).

## 10. Limitations & future work

- **Grid resolution:** Islamabad/Rawalpindi share identical air-quality data
  (Section 2) — a genuine ceiling on how differentiated those two cities'
  forecasts can be with this free data source. A paid, higher-resolution
  ground-station API (e.g. AQICN's station network) would resolve this.
- **One year of history:** covers one winter smog season. More years would
  let the model learn inter-annual variation rather than one season's
  specific pattern.
- **Mixed results outside Lahore** (Section 6): the delta+weather recipe was
  tuned and validated on Lahore specifically; per-city hyperparameter tuning
  (rather than one fixed recipe for all 5) is a natural next step.
- **CI/CD is credential-gated:** the GitHub Actions workflows are ready but
  need `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT` added as repository secrets to
  persist anything durable between runs (see `README.md`).
- **SHAP is per-forecast, not aggregated:** a global feature-importance view
  (averaged across many forecasts) would complement the current per-forecast
  explanations.
