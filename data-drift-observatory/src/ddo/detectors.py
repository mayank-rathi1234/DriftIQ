"""
detectors.py
------------
Five classic distribution-shift statistics, implemented directly (not just
called from a library) so the math is transparent and defensible in an
interview. All functions take a reference (baseline/training) sample and a
current (production batch) sample of a single feature, and return a single
scalar "distance"/"divergence" score. Higher = more drift.

    ks_statistic         Kolmogorov-Smirnov two-sample test statistic
    psi                  Population Stability Index (binned)
    kl_divergence        Kullback-Leibler divergence (binned, symmetrized-safe)
    js_divergence        Jensen-Shannon divergence (binned, bounded [0, ln2])
    wasserstein          Earth Mover's Distance (1D)

For categorical features, PSI, KL and JS are computed directly on category
frequencies; KS and Wasserstein are only defined for ordinal/continuous data
and are skipped for categoricals by the orchestration layer.
"""

import numpy as np
from scipy import stats


def _hist_probs(ref, cur, bins=20, eps=1e-6):
    """Shared binning: build a common bin edge set from the reference
    distribution's range, then histogram both samples into probability
    vectors with a small epsilon floor to avoid log(0)."""
    lo = min(np.min(ref), np.min(cur))
    hi = max(np.max(ref), np.max(cur))
    if lo == hi:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    ref_h, _ = np.histogram(ref, bins=edges)
    cur_h, _ = np.histogram(cur, bins=edges)
    ref_p = ref_h / ref_h.sum() if ref_h.sum() > 0 else np.full(bins, 1 / bins)
    cur_p = cur_h / cur_h.sum() if cur_h.sum() > 0 else np.full(bins, 1 / bins)
    ref_p = ref_p + eps
    cur_p = cur_p + eps
    ref_p /= ref_p.sum()
    cur_p /= cur_p.sum()
    return ref_p, cur_p


def ks_statistic(ref, cur) -> float:
    """Kolmogorov-Smirnov statistic: max distance between empirical CDFs.
    Range [0, 1]. Continuous/ordinal features only."""
    stat, _ = stats.ks_2samp(ref, cur)
    return float(stat)


def psi(ref, cur, bins=10) -> float:
    """Population Stability Index. Industry rule of thumb:
    <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    ref_p, cur_p = _hist_probs(ref, cur, bins=bins)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def kl_divergence(ref, cur, bins=20) -> float:
    """KL(cur || ref): how much information is lost approximating the
    current distribution with the reference one. Not symmetric, unbounded."""
    ref_p, cur_p = _hist_probs(ref, cur, bins=bins)
    return float(np.sum(cur_p * np.log(cur_p / ref_p)))


def js_divergence(ref, cur, bins=20) -> float:
    """Jensen-Shannon divergence: symmetric, bounded in [0, ln(2)],
    generally the most numerically well-behaved of the divergence family."""
    ref_p, cur_p = _hist_probs(ref, cur, bins=bins)
    m = 0.5 * (ref_p + cur_p)
    return float(0.5 * np.sum(ref_p * np.log(ref_p / m)) + 0.5 * np.sum(cur_p * np.log(cur_p / m)))


def wasserstein(ref, cur) -> float:
    """1D Earth Mover's Distance. Unlike KS/KL/JS this is in the same units
    as the feature itself, so it's the most interpretable for "how far did
    this actually move," but it is NOT scale-invariant across features."""
    return float(stats.wasserstein_distance(ref, cur))


def categorical_probs(ref, cur, categories):
    ref_counts = ref.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0)
    eps = 1e-6
    ref_p = (ref_counts / ref_counts.sum()).values + eps
    cur_p = (cur_counts / cur_counts.sum()).values + eps
    ref_p /= ref_p.sum()
    cur_p /= cur_p.sum()
    return ref_p, cur_p


def categorical_psi(ref, cur, categories) -> float:
    ref_p, cur_p = categorical_probs(ref, cur, categories)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def categorical_kl(ref, cur, categories) -> float:
    ref_p, cur_p = categorical_probs(ref, cur, categories)
    return float(np.sum(cur_p * np.log(cur_p / ref_p)))


def categorical_js(ref, cur, categories) -> float:
    ref_p, cur_p = categorical_probs(ref, cur, categories)
    m = 0.5 * (ref_p + cur_p)
    return float(0.5 * np.sum(ref_p * np.log(ref_p / m)) + 0.5 * np.sum(cur_p * np.log(cur_p / m)))


CONTINUOUS_METHODS = {
    "ks": ks_statistic,
    "psi": psi,
    "kl": kl_divergence,
    "js": js_divergence,
    "wasserstein": wasserstein,
}

CATEGORICAL_METHODS = {
    "psi": categorical_psi,
    "kl": categorical_kl,
    "js": categorical_js,
}
