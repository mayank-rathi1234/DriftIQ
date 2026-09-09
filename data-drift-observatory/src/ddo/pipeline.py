"""
pipeline.py
-----------
Ties every module together into the end-to-end flow:

    TRAINING DATA -> baseline profile -> trained model -> PRODUCTION DATA
        -> drift detection (feature / prediction / concept)
        -> root cause analysis
        -> performance prediction
        -> risk score -> retrain decision
        -> Model Health Report

Fixes versus the first version:
  - prediction-drift and concept-drift scalar scores are now normalized
    using only batches <= as_of_batch (no lookahead into future batches).
  - root cause ranking correctly joins raw feature names to model column
    names via train.RAW_TO_MODEL_COL (see root_cause.py).
  - performance forecast reports a real, always-populated backtested MAE
    instead of an often-null "validation" number.
  - the report now includes an explicit retrain decision (retrain_trigger.py)
    instead of only a natural-language recommendation string.
"""

import numpy as np
import pandas as pd

from .scenarios import generate_training_data, simulate_scenario, SCENARIOS, N_FEATURES_CONT
from .train import train_model, encode, MODEL_FEATURE_COLS
from .monitor import build_baseline_profile, run_monitoring
from .concept_drift import run_concept_drift_analysis
from .root_cause import compute_feature_importance, root_cause_ranking
from .performance_predictor import compute_true_accuracy_per_batch, aggregate_drift_score, fit_and_forecast
from .risk_score import compute_risk_score
from .retrain_trigger import should_retrain, notify
from .calibration import smoothed_calibrated_score


def prepare_system(scenario_name: str, n_train=20000, seed=42):
    """One-time setup: train the model, build the baseline profile, and
    simulate the production stream for the requested scenario."""
    train_df = generate_training_data(n=n_train, seed=seed)
    model, device_map, merchant_map = train_model(train_df)
    train_df_enc, _, _ = encode(train_df, device_map, merchant_map)

    cfg = SCENARIOS[scenario_name]
    stream_df = simulate_scenario(cfg)
    stream_df_enc, _, _ = encode(stream_df, device_map, merchant_map)

    baseline = build_baseline_profile(train_df)
    baseline["_ref_X_for_predict"] = train_df_enc[MODEL_FEATURE_COLS].sample(
        min(3000, len(train_df_enc)), random_state=1
    )

    return {
        "train_df": train_df, "train_df_enc": train_df_enc,
        "stream_df": stream_df, "stream_df_enc": stream_df_enc,
        "model": model, "baseline": baseline, "cfg": cfg,
    }


def _causal_normalize(series_by_batch: pd.DataFrame, value_col: str, as_of_batch: int) -> float:
    """Thin wrapper around calibration.smoothed_calibrated_score for
    prediction-drift and concept-drift scalar scores: EWMA-smoothed,
    burn-in-calibrated z-score -- not min-max (proven to falsely inflate
    noise, see test_control_scenario_stays_low_risk) and not a raw
    single-batch z-score either (proven to swing wildly batch-to-batch on
    sustained drift, see the batch-by-batch trend investigation in the
    project history / README)."""
    return smoothed_calibrated_score(series_by_batch, value_col, as_of_batch)


def generate_health_report(system: dict, as_of_batch: int = None) -> dict:
    """Produces the full Model Health Report dict for a given point in time
    (defaults to the most recent batch in the stream)."""
    model = system["model"]
    baseline = system["baseline"]
    train_df_enc = system["train_df_enc"]
    stream_df = system["stream_df"]
    stream_df_enc = system["stream_df_enc"]

    if as_of_batch is None:
        as_of_batch = int(stream_df["batch"].max())

    mon = run_monitoring(baseline, stream_df_enc, model, MODEL_FEATURE_COLS)
    feature_drift_df = mon["feature_drift"]
    prediction_drift_df = mon["prediction_drift"]

    concept_df = run_concept_drift_analysis(
        train_df_enc, stream_df_enc, MODEL_FEATURE_COLS, N_FEATURES_CONT, label_delay=5
    )

    importance = compute_feature_importance(
        model, train_df_enc[MODEL_FEATURE_COLS].sample(2000, random_state=1),
        train_df_enc["is_fraud"].sample(2000, random_state=1), MODEL_FEATURE_COLS
    )
    rc = root_cause_ranking(feature_drift_df, importance, batch=as_of_batch)

    true_acc_df = compute_true_accuracy_per_batch(stream_df_enc, model, MODEL_FEATURE_COLS)
    drift_score_df = aggregate_drift_score(feature_drift_df)  # already causal/expanding internally
    perf = fit_and_forecast(drift_score_df, true_acc_df, as_of_batch, horizons=(5, 10, 20))

    feat_score_now = float(drift_score_df.query("batch == @as_of_batch")["drift_score"].iloc[0])
    pred_score_now = _causal_normalize(prediction_drift_df, "psi", as_of_batch)

    # correlation-shift and MI-shift live on different scales (see
    # concept_drift.py) -- calibrate each independently against its own
    # burn-in noise baseline, THEN take the max, rather than combining
    # raw values first (which previously saturated the score to ~1.0
    # from pure MI-estimation noise regardless of real drift).
    corr_score_now = _causal_normalize(concept_df, "correlation_shift_score", as_of_batch)
    mi_score_now = _causal_normalize(concept_df, "mi_shift_score", as_of_batch)
    concept_score_now = max(corr_score_now, mi_score_now)

    risk = compute_risk_score(feat_score_now, pred_score_now, concept_score_now)
    decision = should_retrain(risk)
    notification = notify(decision, system["cfg"].name)

    report = {
        "scenario": system["cfg"].name,
        "scenario_description": system["cfg"].description,
        "as_of_batch": as_of_batch,
        "risk": risk,
        "drift_breakdown": {
            "feature_drift_score": round(feat_score_now, 3),
            "prediction_drift_score": round(pred_score_now, 3),
            "concept_drift_score": round(concept_score_now, 3),
        },
        "most_affected_features": rc.head(3).to_dict(orient="records"),
        "performance": {
            "current_accuracy": round(perf["current_accuracy"], 4) if perf.get("current_accuracy") is not None else None,
            "forecast_by_batches_ahead": {k: round(v, 4) for k, v in perf["forecast"].items()},
            "backtested_mae_by_horizon": {k: round(v, 4) for k, v in perf["validation"]["mae_by_horizon"].items()},
            "backtested_overall_mae": round(perf["validation"]["overall_mae"], 4) if perf["validation"]["overall_mae"] is not None else None,
        },
        "recommendation": risk["recommendation"],
        "retrain_decision": {
            "should_retrain": decision.should_retrain,
            "urgency": decision.urgency,
            "reason": decision.reason,
        },
        "notification": notification,
    }
    return report, {
        "feature_drift_df": feature_drift_df,
        "prediction_drift_df": prediction_drift_df,
        "concept_df": concept_df,
        "root_cause_df": rc,
        "drift_score_df": drift_score_df,
        "true_acc_df": true_acc_df,
    }
