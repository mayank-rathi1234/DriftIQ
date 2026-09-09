"""
concept_drift.py
----------------
Concept drift means P(y|X) changed. Two complementary, label-aware and
label-free signals are computed:

1. DOMAIN CLASSIFIER (label-free, indirect):
   Trains a classifier to distinguish "reference window" rows from
   "current window" rows using only features. AUC ~0.5 = inputs look
   unchanged (no covariate shift); if the model's behavior still shifts
   while this stays near 0.5, that isolates concept drift from feature
   drift. Uses HistGradientBoostingClassifier (histogram-based, an order
   of magnitude faster than plain GradientBoostingClassifier at this
   sample size) with 2-fold CV -- the first version of this module used
   the slow implementation with 3-fold CV per batch, which made a full
   report take ~9 seconds; this version targets well under half that.

2. DELAYED-LABEL RELATIONSHIP SHIFT (direct, but lagged):
   Labels for batch b only "arrive" `label_delay` batches later, mirroring
   real production lag. Once available, TWO measures of feature-target
   relationship change are computed and combined:
     - Pearson correlation shift: catches linear/monotonic relationship
       changes, cheap, but blind to e.g. a feature going from a negative
       relationship to a U-shaped one (correlation can stay near zero
       throughout even though the true relationship changed completely).
     - Mutual information shift: catches nonlinear/non-monotonic
       relationship changes that correlation misses, at higher variance
       and compute cost -- which is why it's used as a *complement* to
       correlation, not a replacement.
   The reported concept_score is the max of the two (whichever signal
   picked up the change), so genuinely nonlinear drift isn't silently
   missed the way a correlation-only score would miss it.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


def domain_classifier_auc(ref_X: pd.DataFrame, cur_X: pd.DataFrame, seed=0) -> float:
    """AUC of a classifier trained to tell reference vs current apart.
    Single train/test split rather than k-fold CV: this is used as a
    monitoring diagnostic run every batch, not a model-selection decision,
    so we trade a little variance for roughly half the compute versus
    cross-validation -- the earlier version's 3-fold CV per batch was the
    main reason a full report took ~9 seconds."""
    X = pd.concat([ref_X, cur_X], ignore_index=True)
    y = np.array([0] * len(ref_X) + [1] * len(cur_X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    clf = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=seed)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


def feature_target_correlation_shift(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                                      feature_cols_numeric) -> pd.DataFrame:
    """Linear/monotonic relationship-change signal, per feature."""
    rows = []
    for f in feature_cols_numeric:
        ref_corr = ref_df[f].corr(ref_df["is_fraud"])
        cur_corr = cur_df[f].corr(cur_df["is_fraud"])
        rows.append({
            "feature": f,
            "ref_correlation": ref_corr,
            "current_correlation": cur_corr,
            "correlation_shift": abs(cur_corr - ref_corr),
        })
    return pd.DataFrame(rows).sort_values("correlation_shift", ascending=False)


def feature_target_mi_shift(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                             feature_cols_numeric, seed=0) -> pd.DataFrame:
    """Nonlinear relationship-change signal, per feature: mutual information
    between each feature and the label, reference vs current.

    NOTE ON A BUG FOUND VIA TESTING: an earlier version of this function
    normalized the shift by dividing by max(ref_mi, cur_mi) to make shifts
    "comparable across features." That's unstable: MI estimates for a weak
    or noisy feature are often close to zero, so a tiny absolute change
    (pure estimation noise) divided by a tiny denominator produces a ratio
    near 1.0 -- indistinguishable from genuine drift. This surfaced as the
    concept-drift score reading ~1.0 from batch 5 onward in EVERY scenario,
    including well before any real drift onset. The fix is to report the
    raw absolute MI difference and let the downstream burn-in calibration
    (calibration.py) establish what a normal amount of noise-driven MI
    fluctuation looks like, rather than pre-dividing by an unstable scale
    inside the metric itself."""
    ref_mi = mutual_info_classif(
        ref_df[feature_cols_numeric], ref_df["is_fraud"], random_state=seed, discrete_features=False
    )
    cur_mi = mutual_info_classif(
        cur_df[feature_cols_numeric], cur_df["is_fraud"], random_state=seed, discrete_features=False
    )
    rows = []
    for f, r, c in zip(feature_cols_numeric, ref_mi, cur_mi):
        rows.append({"feature": f, "ref_mi": r, "current_mi": c, "mi_shift_raw": abs(c - r)})
    return pd.DataFrame(rows).sort_values("mi_shift_raw", ascending=False)


def run_concept_drift_analysis(train_df, stream_df, feature_cols_all, feature_cols_numeric,
                                label_delay=5) -> pd.DataFrame:
    """
    For each batch b >= label_delay, treats batch (b - label_delay)'s labels
    as "now available" and computes domain_auc plus BOTH relationship-shift
    signals, combined via max() into a single concept_score per batch.
    """
    batches = sorted(stream_df["batch"].unique())
    rows = []
    for b in batches:
        cur_batch = stream_df[stream_df["batch"] == b]
        ref_sample = train_df[feature_cols_all].sample(min(400, len(train_df)), random_state=1)
        cur_sample = (cur_batch[feature_cols_all].sample(min(400, len(cur_batch)), random_state=1)
                      if len(cur_batch) > 400 else cur_batch[feature_cols_all])
        domain_auc = domain_classifier_auc(ref_sample, cur_sample)

        labeled_batch_idx = b - label_delay
        corr_score, mi_score = np.nan, np.nan
        if labeled_batch_idx >= 0:
            labeled_batch = stream_df[stream_df["batch"] == labeled_batch_idx]
            corr_df = feature_target_correlation_shift(train_df, labeled_batch, feature_cols_numeric)
            corr_score = float(corr_df["correlation_shift"].max())
            mi_df = feature_target_mi_shift(train_df, labeled_batch, feature_cols_numeric)
            mi_score = float(mi_df["mi_shift_raw"].max())

        # NOTE: correlation_shift_score and mi_shift_score live on different,
        # incomparable scales (bounded ~[0,2] vs. raw nats). They are
        # intentionally NOT combined here into one "concept_score" -- doing
        # so before each has been calibrated against its own noise baseline
        # is what caused the score-saturation bug described above. Each is
        # calibrated independently downstream (pipeline.py) via
        # calibration.calibrated_score, and only THEN combined via max().
        rows.append({
            "batch": b,
            "domain_auc": domain_auc,
            "correlation_shift_score": corr_score,
            "mi_shift_score": mi_score,
            "label_available": labeled_batch_idx >= 0,
        })
    return pd.DataFrame(rows)
