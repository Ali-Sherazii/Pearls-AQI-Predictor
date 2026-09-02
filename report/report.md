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
are **100% identical** between the two cities across the full year — their
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
- **Circular wind direction:** `wind_direction_10m` (present-time and future)
  is encoded as `sin`/`cos` of its angle rather than passed as a raw degree
  value — a raw degree value tells a model that 359° and 1° (both "almost
  due north") are maximally different, which is wrong and adds noise.
- **Lag/rolling:** `us_aqi` gets lag-1/2/3/6/12/24h, 6h and 24h rolling
  means, and a 24h rolling std (recent volatility); `pm2_5`/`pm10` get
  lag-1/24h and 6h/24h rolling means; the four other pollutants
  (`carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`) get a
  lighter lag-1/24h touch so they contribute more than one noisy
  instantaneous reading each.
- **Derived:** 1h and 24h AQI change rate, and a 3h surface-pressure change
  (a pressure drop often precedes a front moving through that disperses or
  traps pollution — a leading indicator persistence has no way to see).
- **Future weather (serving-time only):** weather forecast at the target
  hour `t+h`, plus cumulative rainfall and mean wind speed between now and
  then — the single biggest driver of forecast accuracy (see below).

This feature set is richer than what experiments/ (Section 5) were run
against — the CV numbers there predate the lag/pressure/wind-direction
additions and describe the *framing* decision (delta vs. absolute, plus
future weather), which the richer feature set doesn't change.

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
serverless dashboard/pipelines don't otherwise need — so Random Forest on
delta + future weather (experiment 3) was the starting point production
built on. Section 6 describes what was added beyond it.

### 5b. Beyond the fixed recipe: per-city model selection + shrinkage

The recipe above was validated on Lahore and initially just reused for every
city, which produced real but uneven results: Lahore beat the persistence
baseline at all 3 horizons, but the other 4 cities only beat it at some
horizons and lost at others (worse than a naive "AQI in h hours = AQI now").
Two changes fixed this, both in `src/models/train.py`:

1. **Model-family selection per city/horizon**, via out-of-fold time-series
   CV predictions over 4 candidates (two Random Forest configs, two
   `HistGradientBoostingRegressor` configs) instead of assuming Lahore's
   tuned Random Forest transfers everywhere. It often doesn't — Lahore and
   Karachi's data favor gradient boosting, Islamabad/Rawalpindi/Peshawar's
   favor Random Forest.
2. **Persistence-shrinkage calibration**: after picking the best family, a
   scalar `alpha` is grid-searched (using the *same* out-of-fold prediction
   pool, so it costs no extra held-out data) so the deployed prediction is
   `current_AQI + alpha * predicted_delta`. `alpha = 0` reproduces
   persistence exactly and is always in the search grid, so calibration can
   never choose something that scores worse than the baseline on that
   pool — this is precisely what fixes the horizons where the raw model's
   delta prediction was directionally right but overconfident in magnitude.
   Calibrated alphas came out at 0.25–0.80 across cities/horizons, meaning
   the raw models were consistently overconfident before shrinkage.

An earlier version of this calibration used a dedicated 20% validation slice
instead of reusing out-of-fold predictions from the train slice. That
version's held-out RMSE was *worse* than the original fixed recipe, despite
still beating the baseline via the alpha=0 guarantee — shrinking the
training slice from 80% to 60% of ~8,760 rows/city cost more real signal
than the extra calibration step gained. Reusing the out-of-fold pool instead
avoids that trade-off entirely.

**Model size:** this is no longer a fixed target — each city/horizon uses
whichever family CV selected. `HistGradientBoostingRegressor` models are
compact (600KB–800KB); the Random Forest configs remain in the
15–45MB range. Total registry size (~265MB across 15 models) is well down
from the original single-recipe exploration (300 trees, unbounded depth,
~180MB *per model*), even though several cities still use Random Forest,
because accuracy (via CV selection) rather than a fixed size cap now
determines the choice.

## 6. Final production results (`src/pipelines/training_pipeline.py`)

Per-city, per-horizon, held out on the most recent 20% of each city's year of
data (never used for model selection or alpha calibration - see Section 5b).
These are the actual models live in the Hopsworks Model Registry as of this
run:

| City | +24h RMSE (base) | +48h RMSE (base) | +72h RMSE (base) |
|---|---|---|---|
| Lahore | 25.04 (34.80) ✅ | 34.07 (42.71) ✅ | 36.11 (42.84) ✅ |
| Islamabad | 18.29 (18.92) ✅ | 23.06 (24.37) ✅ | 25.12 (27.70) ✅ |
| Rawalpindi | 18.10 (18.92) ✅ | 22.58 (24.37) ✅ | 24.34 (27.70) ✅ |
| Karachi | 4.84 (6.17) ✅ | 6.49 (8.73) ✅ | 7.41 (10.22) ✅ |
| Peshawar | 19.09 (20.27) ✅ | 24.17 (25.04) ✅ | 25.88 (26.93) ✅ |

✅ = model beats persistence — **all 15/15 city × horizon combinations now
do**, up from 6/15 with the original fixed Lahore-tuned recipe. R² also
improved across the board (e.g. Islamabad +24h: 0.616, Rawalpindi +24h:
0.624 — both were negative-to-marginal before). Every model is saved to the
registry regardless of whether it beats the baseline — see
`src/pipelines/training_pipeline.py` — so a future regression would be
visible rather than hidden.

**Honest interpretation:** the improvement comes from two independent
sources - the richer feature set (Section 4) gives every model more real
signal to work with, and per-city model selection + shrinkage (Section 5b)
means each city gets whichever family/confidence level actually suits its
data instead of one recipe tuned on Lahore. The persistence-shrinkage
guarantee is exact for the out-of-fold pool it's calibrated against, but not
mathematically guaranteed to hold on the separate, untouched test slice
reported here — it held for all 15 combinations in this run, but a single
scalar recalibrated on a fixed weekly/daily cadence (the training pipeline
already runs daily) is expected to track this reliably rather than
guarantee it in every possible run. (Exact numbers will drift slightly
run-to-run — the tree-based models aren't perfectly reproducible across
different data pulls even with a fixed `random_state`, since the holdout
split point shifts with the row count — but the pattern of beating baseline
at every horizon has been stable across repeated runs.)

## 7. Explainability (`src/models/explain.py`)

Each dashboard forecast is paired with a SHAP `TreeExplainer` breakdown of
the top features driving that specific prediction, scaled by the model's
calibrated shrinkage alpha (Section 5b) so the bars reflect the same units
as the displayed forecast, in AQI-delta units so a positive bar means
"pushes the forecast up from today's AQI." For example, a Lahore +24h
forecast on 2026-09-02 was driven up by `us_aqi`, `pm10`, `aqi_change_24h`,
and `pm2_5`, and driven down by `wind_mean_next24h` and `month_sin` — wind
picking up over the next day was again the single largest downward
contributor, consistent with the EDA's wind/AQI correlation. A Karachi
+24h forecast the same day was driven by different features entirely
(`us_aqi_roll6`, `us_aqi_lag6`, `pm10_lag24`), confirming per-city SHAP
explanations genuinely reflect what that city's own model learned rather
than a shared, generic story.

## 8. Alerts

The dashboard raises a hazard banner whenever any horizon's forecast AQI
reaches 150 (US AQI "Unhealthy for Sensitive Groups" and above) —
`config.HAZARD_AQI`, checked in both `predict`-style serving code and the
dashboard.

## 9. Dashboard (`app/streamlit_app.py`)

City picker across all 5 cities; current AQI with a color-coded category
badge; a 3-day forecast line chart with a hazard-threshold reference line;
a forecast detail table; per-horizon SHAP explanations; and a 14-day AQI
history chart. Verified end-to-end via headless-browser screenshots for
both a high-AQI city (Lahore, current AQI 136 "Unhealthy (sensitive
groups)," forecast staying under the hazard threshold) and a low-AQI city
(Karachi, current AQI 65 "Moderate," no alert), including a correct rerun
on city switch (Streamlit fades the previous city's charts rather than
removing them until the new city's own computation finishes).

## 10. Limitations & future work

- **Grid resolution:** Islamabad/Rawalpindi share identical air-quality data
  (Section 2) — a genuine ceiling on how differentiated those two cities'
  forecasts can be with this free data source. A paid, higher-resolution
  ground-station API (e.g. AQICN's station network) would resolve this.
- **One year of history:** covers one winter smog season. More years would
  let the model learn inter-annual variation rather than one season's
  specific pattern.
- **Shrinkage generalization isn't mathematically guaranteed on new data**
  (Section 5b/6): alpha is calibrated per training run via out-of-fold
  predictions and re-calibrates every time the daily training pipeline runs,
  which should track real drift, but a single anomalous day's data could in
  principle produce a poorly-calibrated alpha before the next run corrects
  it.
- **CI/CD is credential-gated:** the GitHub Actions workflows are ready but
  need `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT` added as repository secrets to
  persist anything durable between runs (see `README.md`).
- **SHAP is per-forecast, not aggregated:** a global feature-importance view
  (averaged across many forecasts) would complement the current per-forecast
  explanations.
