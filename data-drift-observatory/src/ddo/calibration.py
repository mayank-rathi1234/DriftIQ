"""
calibration.py
---------------
Replaces min-max normalization for drift/concept scores with a
baseline-calibrated z-score.

WHY MIN-MAX NORMALIZATION WAS WRONG:
Min-max normalization stretches whatever range of values is observed to
fill exactly [0, 1] -- by construction, the smallest value seen becomes 0
and the largest becomes 1, REGARDLESS of whether the underlying values
represent real drift or pure sampling noise. This meant that even the
"control" scenario (genuinely no injected drift) scored a risk of ~65-90%,
because random noise in a 30-batch window will always have some highest
and lowest points, and min-max normalization dutifully stretches that
noise to look like a near-maximal signal.

THE FIX: burn-in calibration + capped z-score.
The first `reference_batches` batches (default 5) are treated as a
"burn-in" period during which the system learns what NORMAL noise looks
like for a given metric -- mean and standard deviation of the metric under
no assumed drift. Every subsequent batch's score is then a z-score against
that baseline, capped at `z_cap` standard deviations and mapped to [0, 1].
This is the same idea as a statistical process control chart: you don't
call it an anomaly just because it's the batch-relative maximum, you call
it an anomaly because it's meaningfully far from established normal
variation.

Trade-off, stated plainly: this assumes the first few batches are
drift-free. If drift is already present at batch 0 (e.g. a scenario with
onset_batch=0), the baseline itself would be contaminated. None of this
project's scenarios drift before batch 15, but a real deployment should
calibrate against a known-clean historical window, not just "whatever
data arrived first."
"""

import numpy as np
import pandas as pd


def calibrated_score(values_by_batch: pd.DataFrame, value_col: str, as_of_batch: int,
                      reference_batches: int = 5, z_cap: float = 4.0) -> float:
    """Returns a [0,1] score for `value_col` at `as_of_batch`, calibrated
    against the noise level observed in the first `reference_batches`
    batches. 0 = indistinguishable from baseline noise. 1 = at or beyond
    `z_cap` standard deviations above baseline mean."""
    ref = values_by_batch[values_by_batch["batch"] < reference_batches][value_col].dropna()
    if len(ref) < 2:
        # not enough burn-in data yet (e.g. very early batch) -- fall back to
        # all history up to now rather than crashing
        ref = values_by_batch[values_by_batch["batch"] <= as_of_batch][value_col].dropna()
    if len(ref) < 2:
        return 0.0

    mu, sigma = float(ref.mean()), float(ref.std(ddof=0))
    sigma = max(sigma, 1e-9)

    row = values_by_batch[values_by_batch["batch"] == as_of_batch]
    if row.empty or pd.isna(row[value_col].iloc[0]):
        return 0.0

    z = (float(row[value_col].iloc[0]) - mu) / sigma
    return float(np.clip(z / z_cap, 0, 1))


def calibrated_series(values_by_batch: pd.DataFrame, value_col: str,
                       reference_batches: int = 5, z_cap: float = 4.0) -> pd.Series:
    """calibrated_score computed for every batch present, returned as a
    Series indexed by batch. Building the full series is what makes EWMA
    smoothing (smoothed_calibrated_score below) possible."""
    batches = sorted(values_by_batch["batch"].unique())
    vals = {b: calibrated_score(values_by_batch, value_col, b, reference_batches, z_cap) for b in batches}
    return pd.Series(vals).sort_index()


def smoothed_calibrated_score(values_by_batch: pd.DataFrame, value_col: str, as_of_batch: int,
                               reference_batches: int = 5, z_cap: float = 4.0, span: int = 3) -> float:
    """
    WHY THIS EXISTS: a single-batch calibrated_score is a point estimate of
    a noisy statistic (MI on ~600 rows, a domain classifier on a
    train/test split, etc). In practice this produced a risk score that
    swung wildly batch to batch even while the underlying true drift was
    constant and sustained (observed empirically: 60 -> 25 -> 27 -> 12 across
    four consecutive checks on a scenario with no change in ground truth).
    No real monitoring system pages someone off one noisy reading; it
    smooths. This applies an exponentially-weighted moving average (recent
    batches weighted more than older ones, span=3 by default) over the
    full calibrated history up to `as_of_batch`, so a genuinely sustained
    drift stays visible while single-batch noise gets damped, without
    introducing the kind of unbounded lag a simple flat moving average
    would add.
    """
    series = calibrated_series(values_by_batch, value_col, reference_batches, z_cap)
    series = series[series.index <= as_of_batch]
    if series.empty:
        return 0.0
    smoothed = series.ewm(span=span, min_periods=1).mean()
    return float(smoothed.iloc[-1])
