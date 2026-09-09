"""
monitor.py
----------
Orchestrates per-batch drift monitoring: for every production batch, compute
every statistical distance (detectors.py) for every feature against the
fixed reference/baseline window, plus prediction drift on the model's
output distribution. This produces the raw time series that root_cause.py,
concept_drift.py and performance_predictor.py all consume downstream.
"""

import numpy as np
import pandas as pd

from . import detectors as det
from .scenarios import N_FEATURES_CONT, N_FEATURES_CAT, DEVICE_CATS, MERCHANT_CATS

CAT_LEVELS = {"device_type": DEVICE_CATS, "merchant_category": MERCHANT_CATS}


def build_baseline_profile(train_df: pd.DataFrame) -> dict:
    """Freeze the reference distributions the model was trained on.
    In a real system this is computed once at training time and stored
    alongside the model artifact."""
    return {
        "continuous": {f: train_df[f].values for f in N_FEATURES_CONT},
        "categorical": {f: train_df[f] for f in N_FEATURES_CAT},
        "base_rate": float(train_df["is_fraud"].mean()),
    }


def compute_feature_drift(baseline: dict, batch_df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per feature with every distance metric computed
    against the baseline reference distribution for this single batch."""
    rows = []
    for f in N_FEATURES_CONT:
        ref = baseline["continuous"][f]
        cur = batch_df[f].values
        rows.append({
            "feature": f, "type": "continuous",
            "ks": det.ks_statistic(ref, cur),
            "psi": det.psi(ref, cur),
            "kl": det.kl_divergence(ref, cur),
            "js": det.js_divergence(ref, cur),
            "wasserstein": det.wasserstein(ref, cur),
        })
    for f in N_FEATURES_CAT:
        ref = baseline["categorical"][f]
        cur = batch_df[f]
        cats = CAT_LEVELS[f]
        rows.append({
            "feature": f, "type": "categorical",
            "ks": np.nan,
            "psi": det.categorical_psi(ref, cur, cats),
            "kl": det.categorical_kl(ref, cur, cats),
            "js": det.categorical_js(ref, cur, cats),
            "wasserstein": np.nan,
        })
    return pd.DataFrame(rows)


def normalize_metric(series: pd.Series, method: str) -> pd.Series:
    """Different metrics live on different scales (PSI ~0-1+, JS bounded by
    ln2, Wasserstein feature-unit-dependent). To combine them into one score
    we min-max normalize each metric's own history before aggregating."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def run_monitoring(baseline: dict, stream_df: pd.DataFrame, model, feature_cols) -> dict:
    """
    Sweeps every batch in the stream, computing:
      - per-feature, per-metric drift scores  -> feature_drift_df
      - prediction (model output) drift        -> prediction_drift_df
    Returns both as long-format DataFrames indexed by batch.
    """
    batches = sorted(stream_df["batch"].unique())
    feature_rows = []
    pred_rows = []

    ref_X = baseline["_ref_X_for_predict"]
    ref_scores = model.predict_proba(ref_X)[:, 1]

    for b in batches:
        batch_df = stream_df[stream_df["batch"] == b]
        fd = compute_feature_drift(baseline, batch_df)
        fd["batch"] = b
        feature_rows.append(fd)

        cur_scores = model.predict_proba(batch_df[feature_cols])[:, 1]
        pred_rows.append({
            "batch": b,
            "ks": det.ks_statistic(ref_scores, cur_scores),
            "psi": det.psi(ref_scores, cur_scores),
            "js": det.js_divergence(ref_scores, cur_scores),
            "wasserstein": det.wasserstein(ref_scores, cur_scores),
            "mean_pred": float(cur_scores.mean()),
        })

    feature_drift_df = pd.concat(feature_rows, ignore_index=True)
    prediction_drift_df = pd.DataFrame(pred_rows)
    return {"feature_drift": feature_drift_df, "prediction_drift": prediction_drift_df}
