# Pearls AQI Predictor

A 3-day Air Quality Index forecast for 5 Pakistani cities (Lahore, Islamabad,
Rawalpindi, Karachi, Peshawar), built on a serverless feature-store /
model-registry architecture.

```
Open-Meteo API --> feature pipeline --> Feature Store (Hopsworks, or local
                                          parquet fallback)
                                              |
                                              v
                                       training pipeline --> Model Registry
                                              |                (Hopsworks,
                                              v                 or local
                                       Streamlit dashboard <-----joblib)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # optionally add Hopsworks credentials - see below
```

**Windows note:** `hopsworks[python]`'s `pyjks` dependency pulls in `twofish`,
a C extension with no prebuilt Windows wheel, so a plain `pip install -r
requirements.txt` fails there without Visual C++ Build Tools. Easiest real
fix: install "Desktop development with C++" via the
[Build Tools installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
then `pip install -r requirements.txt` works normally. Without Build Tools,
work around it instead - `twofish` is only needed for a keystore cipher this
project never uses:
```bash
pip install --no-deps hopsworks[python]
pip install pyhumps==1.6.1 tzlocal tomli-w sqlalchemy retrying PyMySQL ^
  "avro==1.12.0" "opensearch-py==2.4.2" "protobuf>=4.25.4,<5.0.0" ^
  orderedmultidict mock jmespath furl botocore s3transfer boto3 ^
  confluent-kafka pyarrow fastavro tqdm httpx
pip install --no-deps pyjks pycryptodomex javaobj-py3   # pyjks, minus twofish
pip install -r requirements.txt   # the rest (pandas, streamlit, shap, ...)
```
Linux CI (GitHub Actions' `ubuntu-latest`) has build tools preinstalled, so
`pip install -r requirements.txt` there needs no workaround.

No API key is required for the data source (Open-Meteo). Hopsworks is
optional: without credentials, everything below runs against a local
parquet/joblib fallback under `data/` and `models/` instead - useful for
development, and functionally identical from every other script's point of
view (see `src/feature_store/store.py`, `src/models/registry.py`).

To use real Hopsworks:
1. Create a free account at https://www.hopsworks.ai — this provisions a
   dedicated instance (e.g. `<region>.cloud.hopsworks.ai`) under
   https://run.hopsworks.ai.
2. Click **Access Hopsworks** on that instance, then create a project inside
   it (new accounts start with none) - its name is your `HOPSWORKS_PROJECT`.
3. Inside the instance: profile icon (top-right) -> Settings -> API Keys ->
   New API Key. Give it at least `featurestore`, `project`, and
   `model-registry` scopes, and copy it immediately - it's shown once.
4. Set `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` in `.env` (never commit
   this file - `.env.example` is the tracked template).

## Running the pipelines

```bash
# One-time: backfill a year of history for all 5 cities
python -m src.pipelines.backfill

# Hourly: fetch recent data, engineer features, upsert into the store
python -m src.pipelines.feature_pipeline

# Daily: train + evaluate a model per city/horizon, save to the registry
python -m src.pipelines.training_pipeline
```

Both pipeline scripts accept `--city <name>` (repeatable) to restrict to a
subset of cities; omit it to run all 5.

## Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

## CI/CD

`.github/workflows/feature_pipeline.yml` and `training_pipeline.yml` run the
pipelines above on a schedule (hourly / daily). They need `HOPSWORKS_API_KEY`
and `HOPSWORKS_PROJECT` set as repository secrets to persist anything useful
between runs (GitHub Actions runners are ephemeral, so the local-fallback
mode has nowhere durable to write) - add them under
Settings -> Secrets and variables -> Actions.

## Project layout

- `config.py` - cities, API endpoints, forecast horizons, Hopsworks naming
- `src/data/fetch.py` - Open-Meteo air-quality + weather fetchers
- `src/features/engineering.py` - the one feature-engineering implementation
  shared by backfill, the hourly pipeline, training, and serving
- `src/feature_store/`, `src/models/registry.py` - Hopsworks-backed store and
  registry, each with a local fallback
- `src/models/train.py` - the winning recipe (Random Forest on AQI-delta +
  forecast weather, see `experiments/`) generalized across cities/horizons
- `src/models/explain.py` - SHAP feature-contribution explanations
- `src/pipelines/` - the three pipeline entry points above
- `app/streamlit_app.py` - the dashboard
- `experiments/` - the model-selection history: `train_cv.py` (persistence
  vs Ridge vs Random Forest) -> `train_delta_cv.py` (delta framing) ->
  `train_weather_cv.py` (+ forecast weather, the winning recipe) ->
  `train_nn_cv.py` (+ a deep-learning candidate)
- `notebooks/01_eda.ipynb` - cross-city exploratory data analysis
- `report/report.md` - full writeup

## Reproducing the experiments

```bash
python experiments/train_cv.py
python experiments/train_delta_cv.py
python experiments/train_weather_cv.py
pip install tensorflow-cpu   # only needed for this one
python experiments/train_nn_cv.py
```
