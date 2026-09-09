"""
Basic sanity tests: run with `pytest tests/`.
These check the statistical primitives behave as expected (e.g. identical
distributions -> ~0 divergence; clearly shifted distributions -> large
divergence) rather than testing exact values, since exact values depend on
random seeds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from ddo import detectors as det


def test_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)
    assert det.ks_statistic(ref, cur) < 0.05
    assert det.psi(ref, cur) < 0.05
    assert det.js_divergence(ref, cur) < 0.02
    assert det.wasserstein(ref, cur) < 0.1


def test_shifted_distribution_detected():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)
    assert det.ks_statistic(ref, cur) > 0.5
    assert det.psi(ref, cur) > 0.25
    assert det.wasserstein(ref, cur) > 2.0


def test_js_divergence_bounded():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 3000)
    cur = rng.uniform(-10, 10, 3000)
    js = det.js_divergence(ref, cur)
    assert 0 <= js <= np.log(2) + 1e-6


def test_categorical_psi_zero_when_identical():
    import pandas as pd
    ref = pd.Series(["a", "b", "c"] * 1000)
    cur = pd.Series(["a", "b", "c"] * 1000)
    assert det.categorical_psi(ref, cur, ["a", "b", "c"]) < 1e-3
