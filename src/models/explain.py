"""
SHAP explanations for a single forecast row - "what's driving this number."
"""

from __future__ import annotations
import pandas as pd
import shap


def top_feature_contributions(bundle: dict, row: dict, top_n: int = 8) -> pd.Series:
    """Signed SHAP contribution (in AQI-delta units, i.e. already scaled by
    the model's calibrated shrinkage alpha - see src/models/train.py) for the
    top_n features driving one prediction. Positive = pushes the forecast up,
    negative = down."""
    feature_cols = bundle["feature_cols"]
    x = pd.DataFrame([row]).reindex(columns=feature_cols)

    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(x)

    contrib = pd.Series(shap_values[0], index=feature_cols) * bundle.get("alpha", 1.0)
    order = contrib.abs().sort_values(ascending=False).index
    return contrib.reindex(order).head(top_n)
