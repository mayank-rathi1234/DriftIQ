"""
scenarios.py
------------
Generates a synthetic "transaction fraud" style binary classification
dataset and simulates a stream of production batches under six distinct
drift regimes. Each scenario carries a KNOWN ground-truth drift onset
point, which is what lets us benchmark detectors against reality later
(benchmark.py) instead of just eyeballing plots.

Features (mirrors a realistic fraud/credit use case):
    transaction_amount   (continuous, log-normal)
    customer_age         (continuous, normal)
    account_tenure_days  (continuous, exponential)
    device_type          (categorical: mobile/desktop/tablet)
    merchant_category    (categorical: 6 categories)
    hour_of_day          (continuous, cyclic 0-23)

Target:
    is_fraud (binary), generated from a fixed logistic function of the
    features so we can precisely control when/how P(y|X) changes.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


N_FEATURES_CONT = ["transaction_amount", "customer_age", "account_tenure_days", "hour_of_day"]
N_FEATURES_CAT = ["device_type", "merchant_category"]
DEVICE_CATS = ["mobile", "desktop", "tablet"]
MERCHANT_CATS = ["grocery", "electronics", "travel", "dining", "fuel", "online_retail"]


@dataclass
class ScenarioConfig:
    name: str
    description: str
    n_batches: int = 30
    batch_size: int = 1500
    onset_batch: int = 15          # batch index where drift begins (ground truth)
    seed: int = 42


def _base_features(n, rng):
    """Draw features from the reference/training distribution."""
    transaction_amount = rng.lognormal(mean=3.2, sigma=0.9, size=n)
    customer_age = rng.normal(loc=41, scale=13, size=n).clip(18, 90)
    account_tenure_days = rng.exponential(scale=650, size=n).clip(0, 5000)
    hour_of_day = rng.uniform(0, 24, size=n)
    device_type = rng.choice(DEVICE_CATS, size=n, p=[0.55, 0.35, 0.10])
    merchant_category = rng.choice(MERCHANT_CATS, size=n,
                                    p=[0.25, 0.15, 0.10, 0.20, 0.15, 0.15])
    return pd.DataFrame({
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "account_tenure_days": account_tenure_days,
        "hour_of_day": hour_of_day,
        "device_type": device_type,
        "merchant_category": merchant_category,
    })


def _label_fn(df, concept_shift=0.0, rng=None):
    """
    Fixed logistic fraud-generating process.
    `concept_shift` rotates the decision boundary by changing the
    coefficient on transaction_amount relative to customer_age --
    this is what makes Scenario C genuine concept drift (P(y|X) change)
    rather than just covariate shift.
    """
    z = (
        -6.0
        + (0.9 + concept_shift) * np.log1p(df["transaction_amount"])
        - (0.03 - concept_shift * 0.02) * df["customer_age"]
        - 0.0015 * df["account_tenure_days"]
        + 0.35 * (df["device_type"] == "mobile").astype(float)
        + 0.5 * (df["merchant_category"] == "online_retail").astype(float)
        + 0.15 * np.sin(df["hour_of_day"] / 24 * 2 * np.pi)
    )
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    return y, p


def generate_training_data(n=20000, seed=42) -> pd.DataFrame:
    """The reference/baseline dataset the model is trained on."""
    rng = np.random.default_rng(seed)
    df = _base_features(n, rng)
    y, p_true = _label_fn(df, concept_shift=0.0, rng=rng)
    df["is_fraud"] = y
    return df


def _apply_drift(df, scenario: str, batch_idx: int, cfg: ScenarioConfig, rng):
    """Mutates a batch's features/labels in place per scenario semantics.
    Returns (df, concept_shift) -- concept_shift feeds into _label_fn.
    """
    progressed = batch_idx >= cfg.onset_batch
    concept_shift = 0.0

    if scenario == "control":
        return df, concept_shift

    if scenario == "A_customer_population":
        # Sudden shift in categorical mix: device + merchant proportions change
        # (new, younger, more mobile-first customer segment enters)
        if progressed:
            df["device_type"] = rng.choice(DEVICE_CATS, size=len(df), p=[0.80, 0.12, 0.08])
            df["customer_age"] = rng.normal(loc=27, scale=7, size=len(df)).clip(18, 90)

    elif scenario == "B_feature_distribution":
        # Sudden shift in a continuous feature's distribution (e.g. pricing
        # tier change inflates transaction amounts)
        if progressed:
            df["transaction_amount"] = df["transaction_amount"] * 2.4 + rng.normal(0, 15, len(df))
            df["transaction_amount"] = df["transaction_amount"].clip(lower=1)

    elif scenario == "C_relationship_change":
        # TRUE concept drift: P(y|X) changes, marginal P(X) barely moves
        if progressed:
            concept_shift = 0.55

    elif scenario == "D_sudden_outliers":
        # Point contamination injected only in a narrow window
        if cfg.onset_batch <= batch_idx < cfg.onset_batch + 2:
            n_out = max(1, int(0.06 * len(df)))
            idx = rng.choice(len(df), size=n_out, replace=False)
            df.loc[idx, "transaction_amount"] = rng.uniform(5000, 20000, size=n_out)

    elif scenario == "E_gradual_drift":
        # Slow linear drift starting at onset, ramping over remaining batches
        if progressed:
            t = (batch_idx - cfg.onset_batch) / max(1, (cfg.n_batches - cfg.onset_batch))
            df["account_tenure_days"] = df["account_tenure_days"] * (1 - 0.6 * t)
            df["customer_age"] = df["customer_age"] - 10 * t

    elif scenario == "F_seasonal_drift":
        # Recurring/periodic drift (e.g. weekly cycle) -- present from day 1,
        # onset_batch here marks when amplitude becomes large enough to matter
        phase = 2 * np.pi * (batch_idx % 7) / 7
        amplitude = 0.15 if batch_idx < cfg.onset_batch else 0.55
        df["transaction_amount"] = df["transaction_amount"] * (1 + amplitude * np.sin(phase))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return df, concept_shift


def simulate_scenario(cfg: ScenarioConfig) -> pd.DataFrame:
    """
    Produces a long-format DataFrame with one row per transaction, tagged
    with `batch` (simulated day index) and `is_drift_truth` (ground-truth
    label for whether this batch is post-drift-onset) so detectors can be
    scored objectively later.
    """
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for b in range(cfg.n_batches):
        batch_df = _base_features(cfg.batch_size, rng)
        batch_df, concept_shift = _apply_drift(batch_df, cfg.name, b, cfg, rng)
        y, _ = _label_fn(batch_df, concept_shift=concept_shift, rng=rng)
        batch_df["is_fraud"] = y
        batch_df["batch"] = b
        batch_df["is_drift_truth"] = int(b >= cfg.onset_batch)
        frames.append(batch_df)
    return pd.concat(frames, ignore_index=True)


SCENARIOS = {
    "control": ScenarioConfig("control", "No drift (negative control)"),
    "A_customer_population": ScenarioConfig(
        "A_customer_population", "Customer distribution changes (new segment)"),
    "B_feature_distribution": ScenarioConfig(
        "B_feature_distribution", "Feature distribution changes (amount inflation)"),
    "C_relationship_change": ScenarioConfig(
        "C_relationship_change", "Relationship between X and Y changes (true concept drift)"),
    "D_sudden_outliers": ScenarioConfig(
        "D_sudden_outliers", "Sudden outlier contamination"),
    "E_gradual_drift": ScenarioConfig(
        "E_gradual_drift", "Gradual drift ramping in slowly"),
    "F_seasonal_drift": ScenarioConfig(
        "F_seasonal_drift", "Seasonal / periodic drift"),
}
