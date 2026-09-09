"""
performance_predictor.py
-------------------------
Forecasts future model performance from the observed drift-metric trend.

Two fixes versus the first version of this module:

1. NO LOOKAHEAD, AND NO MIN-MAX NOISE INFLATION: `aggregate_drift_score`
   now calibrates each metric against a burn-in reference window (see
   calibration.py) instead of min-max normalizing across history. Two
   separate problems are fixed by this one change: (a) min-max used the
   FULL history including future batches (a lookahead leak), and (b) even
   a causal/expanding min-max would still guarantee that pure noise gets
   stretched to look like near-maximal drift at whichever batch happens to
   be locally highest -- which is exactly what caused the "control"
   (no-drift) scenario to falsely score ~65-90% risk in the first version.

2. HONEST, ALWAYS-POPULATED VALIDATION: the previous version tried to
   validate a forecast against "future" batches that frequently didn't
   exist yet in the stream, so `validation_mae` came back `None` most of
   the time. This version backtests properly: for each horizon h, it
   refits using ONLY data up to (as_of_batch - h), forecasts h steps
   forward to land exactly on as_of_batch, and compares against the
   already-known true accuracy at as_of_batch. This is standard
   walk-forward validation and produces a real error estimate on every
   call (once enough history exists), rather than an optimistic guess
   about the future.

The forward-looking "forecast" reported to the user is a SEPARATE fit
using all data up to as_of_batch, extrapolated beyond it -- that part is
never validated directly (it can't be, until the future arrives), and the
backtested MAE above is reported alongside it specifically so the honest
uncertainty is visible instead of implied away.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from .calibration import calibrated_score, smoothed_calibrated_score


def compute_true_accuracy_per_batch(stream_df, model, feature_cols) -> pd.DataFrame:
    rows = []
    for b in sorted(stream_df["batch"].unique()):
        batch_df = stream_df[stream_df["batch"] == b]
        preds = model.predict(batch_df[feature_cols])
        acc = accuracy_score(batch_df["is_fraud"], preds)
        rows.append({"batch": b, "true_accuracy": acc})
    return pd.DataFrame(rows)


def aggregate_drift_score(feature_drift_df: pd.DataFrame, reference_batches: int = 5, span: int = 3) -> pd.DataFrame:
    """Single scalar drift score per batch in [0, 1]: for each of
    PSI/JS/KS, pool across all features, calibrate each batch's mean value
    against the burn-in reference window, EWMA-smooth (span=3) to damp
    single-batch noise the same way pipeline._causal_normalize does for
    prediction/concept drift, then average the three smoothed scores.
    This is causal (no future batches), noise-resistant (calibrated
    against a baseline, not min-max), and stable (smoothed) -- all three
    fixes needed after the original min-max version was shown to falsely
    flag the no-drift control scenario as high risk."""
    df = feature_drift_df.copy()
    pooled = df.groupby("batch")[["psi", "js", "ks"]].mean().reset_index()
    batches = sorted(pooled["batch"].unique())

    scores = []
    for b in batches:
        sub_scores = [
            smoothed_calibrated_score(pooled, m, b, reference_batches=reference_batches, span=span)
            for m in ["psi", "js", "ks"]
        ]
        scores.append({"batch": b, "drift_score": float(np.mean(sub_scores))})
    return pd.DataFrame(scores)


def _fit_ridge(hist: pd.DataFrame) -> Ridge:
    reg = Ridge(alpha=1.0)
    reg.fit(hist[["batch", "drift_score"]].values, hist["true_accuracy"].values)
    return reg


def _extrapolate_drift(hist: pd.DataFrame, steps: int) -> float:
    recent = hist.tail(5)
    slope = np.polyfit(recent["batch"], recent["drift_score"], 1)[0] if len(recent) >= 2 else 0.0
    return float(np.clip(hist["drift_score"].iloc[-1] + slope * steps, 0, 1))


def backtest_validate(merged: pd.DataFrame, as_of_batch: int, horizons) -> dict:
    """Walk-forward validation: for each horizon, refit using only data
    that would genuinely have been available h batches before as_of_batch,
    forecast forward to as_of_batch, and compare to the now-known truth."""
    errors_by_horizon = {}
    for h in horizons:
        cutoff = as_of_batch - h
        train_hist = merged.query("batch <= @cutoff")
        truth_row = merged.query("batch == @as_of_batch")
        if len(train_hist) < 5 or truth_row.empty:
            continue
        reg = _fit_ridge(train_hist)
        future_drift = _extrapolate_drift(train_hist, h)
        pred_acc = float(np.clip(reg.predict([[as_of_batch, future_drift]])[0], 0, 1))
        true_acc = float(truth_row["true_accuracy"].iloc[0])
        errors_by_horizon[h] = abs(true_acc - pred_acc)
    overall_mae = float(np.mean(list(errors_by_horizon.values()))) if errors_by_horizon else None
    return {"mae_by_horizon": errors_by_horizon, "overall_mae": overall_mae}


def fit_and_forecast(drift_score_df: pd.DataFrame, true_acc_df: pd.DataFrame,
                      as_of_batch: int, horizons=(5, 10, 20)) -> dict:
    merged = drift_score_df.merge(true_acc_df, on="batch")
    hist = merged.query("batch <= @as_of_batch")
    if len(hist) < 5:
        return {"forecast": {}, "validation": {"mae_by_horizon": {}, "overall_mae": None}, "current_accuracy": None}

    # forward-looking forecast: fit on everything known so far, extrapolate beyond it
    reg = _fit_ridge(hist)
    forecast = {}
    for h in horizons:
        future_batch = hist["batch"].max() + h
        future_drift = _extrapolate_drift(hist, h)
        pred_acc = float(np.clip(reg.predict([[future_batch, future_drift]])[0], 0, 1))
        forecast[h] = pred_acc

    # honest validation via walk-forward backtesting (always populated when enough history exists)
    validation = backtest_validate(merged, as_of_batch, horizons)

    return {
        "forecast": forecast,
        "validation": validation,
        "as_of_batch": as_of_batch,
        "current_accuracy": float(hist["true_accuracy"].iloc[-1]),
    }
