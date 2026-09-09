"""
benchmark.py
------------
The research core of the project: answers "which drift-detection methods
are most reliable under different types and magnitudes of distribution
shift?" using the ground-truth drift onset baked into every scenario
(scenarios.py sets `is_drift_truth` per batch).

For every scenario x every statistical method x every feature, we compute:

  - AUROC: treating "is this batch post-onset" as the binary label and the
    metric's value as the score, how well does the metric alone separate
    pre/post drift batches? (1.0 = perfect separation, 0.5 = useless)
  - Detection lag: using a data-driven threshold (mean + 3*std of the
    metric's PRE-onset values), how many batches after the true onset does
    the metric first cross that threshold? (lower = faster detection)
  - False positive rate: fraction of PRE-onset batches that incorrectly
    cross the same threshold.

This produces a real, defensible leaderboard -- not asserted, computed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .scenarios import SCENARIOS, generate_training_data
from .train import train_model, encode, MODEL_FEATURE_COLS
from .monitor import build_baseline_profile, compute_feature_drift
from .scenarios import simulate_scenario

METHODS = ["ks", "psi", "kl", "js", "wasserstein"]


def _threshold_detection_lag(values: pd.Series, truth: pd.Series):
    pre = values[truth == 0]
    if len(pre) < 3 or pre.isna().all():
        return np.nan, np.nan
    threshold = pre.mean() + 3 * pre.std(ddof=0)
    post = values[truth == 1]
    onset_idx = truth[truth == 1].index.min()
    crossed = values[(values.index >= onset_idx) & (values > threshold)]
    lag = (crossed.index.min() - onset_idx) if len(crossed) else np.nan
    fpr = float((pre > threshold).mean())
    return lag, fpr


def run_benchmark(n_train=8000, seed=42) -> pd.DataFrame:
    train_df = generate_training_data(n=n_train, seed=seed)
    model, device_map, merchant_map = train_model(train_df)
    baseline = build_baseline_profile(train_df)

    rows = []
    for name, cfg in SCENARIOS.items():
        if name == "control":
            continue
        stream_df = simulate_scenario(cfg)
        batches = sorted(stream_df["batch"].unique())

        # compute feature drift per batch
        fd_frames = []
        for b in batches:
            batch_df = stream_df[stream_df["batch"] == b]
            fd = compute_feature_drift(baseline, batch_df)
            fd["batch"] = b
            fd["is_drift_truth"] = batch_df["is_drift_truth"].iloc[0]
            fd_frames.append(fd)
        fd_all = pd.concat(fd_frames, ignore_index=True)

        for feature in fd_all["feature"].unique():
            fdf = fd_all[fd_all["feature"] == feature].set_index("batch").sort_index()
            truth = fdf["is_drift_truth"]
            if truth.nunique() < 2:
                continue
            for method in METHODS:
                vals = fdf[method]
                if vals.isna().all():
                    continue
                try:
                    auc = roc_auc_score(truth, vals.fillna(vals.mean()))
                except ValueError:
                    auc = np.nan
                lag, fpr = _threshold_detection_lag(vals, truth)
                rows.append({
                    "scenario": name, "feature": feature, "method": method,
                    "auroc": auc, "detection_lag_batches": lag, "false_positive_rate": fpr,
                })
    return pd.DataFrame(rows)


def summarize_benchmark(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates to the scenario x method level (mean across features) --
    this is the headline leaderboard table for the README/report."""
    return (
        bench_df.groupby(["scenario", "method"])
        .agg(mean_auroc=("auroc", "mean"),
             mean_detection_lag=("detection_lag_batches", "mean"),
             mean_fpr=("false_positive_rate", "mean"))
        .reset_index()
        .sort_values(["scenario", "mean_auroc"], ascending=[True, False])
    )


if __name__ == "__main__":
    bench = run_benchmark()
    bench.to_csv("results/benchmark_raw.csv", index=False)
    summary = summarize_benchmark(bench)
    summary.to_csv("results/benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))
