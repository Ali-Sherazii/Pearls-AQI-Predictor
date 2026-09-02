"""
Pearls AQI Predictor dashboard.

Loads features from the feature store and models from the model registry
(src/feature_store, src/models/registry - Hopsworks-backed, or the local
fallback if Hopsworks isn't configured), computes a live 3-day forecast the
same way src/pipelines does, and explains each forecast with SHAP.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config/src

from config import CITIES, FORECAST_HORIZONS_H, HAZARD_AQI
from src.data.fetch import fetch_forecast_weather
from src.features.engineering import future_weather_at
from src.feature_store.store import read_features
from src.models.registry import load_model
from src.models.explain import top_feature_contributions

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

CATEGORY_BREAKS = [
    (50, "Good", "#2ecc71"),
    (100, "Moderate", "#f1c40f"),
    (150, "Unhealthy (sensitive groups)", "#e67e22"),
    (200, "Unhealthy", "#e74c3c"),
    (300, "Very Unhealthy", "#8e44ad"),
    (10 ** 9, "Hazardous", "#7d3c98"),
]


def category(aqi: float) -> tuple[str, str]:
    for hi, name, color in CATEGORY_BREAKS:
        if aqi <= hi:
            return name, color
    return "Hazardous", "#7d3c98"


@st.cache_data(ttl=300, show_spinner=False)
def load_features(city: str) -> pd.DataFrame:
    return read_features(city)


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast_weather(city: str) -> pd.DataFrame:
    return fetch_forecast_weather(city)


@st.cache_resource(show_spinner=False)
def load_model_cached(city: str, horizon_h: int) -> dict:
    return load_model(city, horizon_h)


st.title("🌫️ Pearls AQI Predictor")
st.caption("3-day Air Quality Index forecast for Pakistani cities, 100% serverless")

city = st.selectbox("City", list(CITIES), format_func=str.title)

try:
    features = load_features(city)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

now_row = features.iloc[-1]
T = features.index[-1]
current_aqi = float(now_row["us_aqi"])
cat_name, cat_color = category(current_aqi)

col1, col2 = st.columns([1, 2])
with col1:
    st.metric(f"{city.title()} — current US AQI", f"{current_aqi:.0f}")
    st.markdown(
        f"<div style='background:{cat_color};padding:10px;border-radius:6px;"
        f"color:white;text-align:center;font-weight:600'>{cat_name}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"as of {T:%Y-%m-%d %H:%M}")

with st.spinner("Fetching forecast weather..."):
    fcw = load_forecast_weather(city)

rows, hazards, explanations = [], [], {}
for h in FORECAST_HORIZONS_H:
    try:
        bundle = load_model_cached(city, h)
    except FileNotFoundError as e:
        st.warning(str(e))
        continue

    fw = future_weather_at(fcw, T, h)
    row = {**now_row.to_dict(), **fw}
    x = pd.DataFrame([row]).reindex(columns=bundle["feature_cols"])
    if x.isna().any().any():
        missing = x.columns[x.isna().any()].tolist()
        st.warning(f"+{h}h: missing forecast features {missing[:5]}... skipping.")
        continue

    delta = bundle["model"].predict(x)[0]
    aqi = current_aqi + bundle.get("alpha", 1.0) * delta
    cname, _ = category(aqi)
    target_time = T + timedelta(hours=h)
    rows.append({"horizon_h": f"+{h}h", "time": target_time, "aqi": aqi, "category": cname})
    if aqi >= HAZARD_AQI:
        hazards.append(h)
    explanations[h] = top_feature_contributions(bundle, row)

if hazards:
    st.error(
        f"⚠ ALERT: hazardous AQI (≥ {HAZARD_AQI}) forecast at "
        + ", ".join(f"+{h}h" for h in hazards)
    )

forecast_df = pd.DataFrame(rows)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[T] + list(forecast_df["time"]),
        y=[current_aqi] + list(forecast_df["aqi"]),
        mode="lines+markers", name="Forecast AQI",
    ))
    fig.add_hline(y=HAZARD_AQI, line_dash="dash", line_color="red",
                  annotation_text="Hazard threshold")
    fig.update_layout(title="3-day AQI forecast", yaxis_title="US AQI", xaxis_title="Time")
    st.plotly_chart(fig, width='stretch')

st.subheader("Forecast detail")
st.dataframe(
    forecast_df.style.format({"aqi": "{:.0f}"}),
    width='stretch', hide_index=True,
)

if explanations:
    st.subheader("What's driving each forecast (SHAP)")
    tabs = st.tabs([f"+{h}h" for h in explanations])
    for tab, h in zip(tabs, explanations):
        with tab:
            contrib = explanations[h].sort_values()
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib.values]
            fig2 = go.Figure(go.Bar(
                x=contrib.values, y=contrib.index, orientation="h",
                marker_color=colors,
            ))
            fig2.update_layout(
                title=f"Top features for +{h}h forecast (SHAP, AQI-delta units)",
                xaxis_title="Contribution to predicted change (red = pushes AQI up)",
            )
            st.plotly_chart(fig2, width='stretch')

st.subheader("Recent AQI history")
hist = features.tail(24 * 14)  # last 14 days
fig3 = go.Figure(go.Scatter(x=hist.index, y=hist["us_aqi"], mode="lines", name="US AQI"))
fig3.update_layout(title=f"{city.title()} — last 14 days", yaxis_title="US AQI")
st.plotly_chart(fig3, width='stretch')
