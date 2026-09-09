"""
root_cause.py
-------------
Turns "which features drifted" into "which features are most LIKELY
responsible for the model degrading":

  1. Normalize each feature's drift metrics (0-1 scale) using ONLY
     history up to and including the batch being reported on -- normalizing
     against the full dataset (including future batches) would leak
     information a real production system would not have "today." This
     was a lookahead bug in the first version; fixed here.
  2. Average across the 5 statistical methods for a single per-feature
     drift severity in [0, 1].
  3. Weight severity by the feature's importance to the model. Raw drift
     tables are keyed by human-readable feature names (e.g. "device_type"),
     while model importance is keyed by the model's actual input columns
     (e.g. "device_type_enc"). These must be joined through
     train.RAW_TO_MODEL_COL, not assumed to match -- an earlier version of
     this project silently zeroed out every categorical feature's
     importance because of this exact mismatch.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .train import RAW_TO_MODEL_COL


def compute_feature_importance(model, X_ref, y_ref, feature_cols, n_repeats=5, seed=0) -> pd.Series:
    """Returns a Series indexed by MODEL column names (e.g. 'device_type_enc')."""
    result = permutation_importance(
        model, X_ref, y_ref, n_repeats=n_repeats, random_state=seed, scoring="roc_auc"
    )
    return pd.Series(result.importances_mean, index=feature_cols).clip(lower=0)


def root_cause_ranking(feature_drift_df: pd.DataFrame, importance: pd.Series, batch: int) -> pd.DataFrame:
    """
    feature_drift_df: long-format output of monitor.compute_feature_drift,
    concatenated across all batches (needs a 'batch' column), keyed by RAW
    feature names.
    importance: Series keyed by MODEL column names (from
    compute_feature_importance), translated internally via RAW_TO_MODEL_COL.
    Returns features ranked by (normalized drift severity x model importance)
    for the requested batch, using only data up to `batch` (no lookahead).
    """
    metrics = ["ks", "psi", "kl", "js", "wasserstein"]
    df = feature_drift_df[feature_drift_df["batch"] <= batch].copy()

    # normalize each metric across history UP TO THIS BATCH ONLY
    norm_cols = []
    for m in metrics:
        col = f"{m}_norm"
        vals = df[m]
        lo, hi = vals.min(skipna=True), vals.max(skipna=True)
        if pd.isna(lo) or hi - lo < 1e-9:
            df[col] = 0.0
        else:
            df[col] = ((vals - lo) / (hi - lo)).fillna(0.0)
        norm_cols.append(col)

    df["severity"] = df[norm_cols].mean(axis=1)

    batch_df = df[df["batch"] == batch].copy()
    # translate raw feature name -> model column name to look up importance correctly
    batch_df["model_col"] = batch_df["feature"].map(RAW_TO_MODEL_COL)
    batch_df["importance"] = batch_df["model_col"].map(importance).fillna(0.0)

    imp_max = batch_df["importance"].max()
    batch_df["importance_norm"] = batch_df["importance"] / imp_max if imp_max > 0 else 0.0
    batch_df["root_cause_score"] = batch_df["severity"] * (0.5 + 0.5 * batch_df["importance_norm"])

    return batch_df[["feature", "severity", "importance_norm", "root_cause_score"]] \
        .sort_values("root_cause_score", ascending=False) \
        .reset_index(drop=True)
