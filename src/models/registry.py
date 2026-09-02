"""
Model registry: Hopsworks-backed, with a local-joblib fallback - mirrors
src/feature_store/store.py's dual-mode approach exactly.
"""

from __future__ import annotations
import tempfile
from pathlib import Path

import joblib

from config import MODEL_DIR, HOPSWORKS_ENABLED, MODEL_REGISTRY_VERSION, model_name
from src.feature_store.store import get_hopsworks_project


def _local_path(city: str, horizon_h: int) -> Path:
    return MODEL_DIR / f"{city}_t{horizon_h}h.joblib"


def save_model(city: str, horizon_h: int, bundle: dict, metrics: dict) -> Path:
    local_path = _local_path(city, horizon_h)
    joblib.dump(bundle, local_path)

    if HOPSWORKS_ENABLED:
        project = get_hopsworks_project()
        mr = project.get_model_registry()
        model = mr.python.create_model(
            name=model_name(city, horizon_h),
            metrics=metrics,
            description=f"AQI +{horizon_h}h RF-delta model for {city}",
        )
        model.save(str(local_path))

    return local_path


def load_model(city: str, horizon_h: int) -> dict:
    if HOPSWORKS_ENABLED:
        project = get_hopsworks_project()
        mr = project.get_model_registry()
        model = mr.get_model(model_name(city, horizon_h), version=MODEL_REGISTRY_VERSION)
        # Hopsworks' own download cache is keyed by model id and is supposed
        # to auto-invalidate when a version is deleted/recreated, but that
        # didn't hold up in practice (a retrain under the same name/version
        # served a stale cached copy). Downloading to a fresh temp dir every
        # call sidesteps that entirely - cheap given these models are a few
        # MB, and the dashboard's own @st.cache_resource already avoids
        # repeat downloads within one running process.
        model_dir = Path(model.download(local_path=tempfile.mkdtemp(prefix="aqi_model_")))
        path = next(model_dir.glob("*.joblib"))
        return joblib.load(path)

    local_path = _local_path(city, horizon_h)
    if not local_path.exists():
        raise FileNotFoundError(
            f"No local model for {city!r} +{horizon_h}h - run the training pipeline first."
        )
    return joblib.load(local_path)
