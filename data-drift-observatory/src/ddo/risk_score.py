"""
risk_score.py
-------------
Combines feature drift, prediction drift, and concept drift signals into a
single 0-100 "Model Risk Score" for the health report -- and a plain
recommendation string. Weights are a documented, adjustable design choice,
not a black box:

    feature drift    x 0.25   (covariate shift -- may or may not hurt the model)
    prediction drift x 0.30   (output distribution shift -- more directly tied to behavior change)
    concept drift    x 0.45   (P(y|X) change -- the most dangerous kind, weighted highest)

Each sub-score is itself already normalized to [0, 1] upstream; this module
just does the weighted blend + thresholding into a recommendation.
"""

import numpy as np

WEIGHTS = {"feature": 0.25, "prediction": 0.30, "concept": 0.45}


def compute_risk_score(feature_drift_score: float, prediction_drift_score: float,
                        concept_drift_score: float) -> dict:
    feature_drift_score = float(np.clip(feature_drift_score, 0, 1))
    prediction_drift_score = float(np.clip(prediction_drift_score, 0, 1))
    concept_drift_score = float(np.clip(concept_drift_score, 0, 1))

    blended = (
        WEIGHTS["feature"] * feature_drift_score
        + WEIGHTS["prediction"] * prediction_drift_score
        + WEIGHTS["concept"] * concept_drift_score
    )
    risk_0_100 = round(blended * 100, 1)

    if risk_0_100 >= 70:
        level, recommendation = "critical", "Retrain immediately using recent production data."
    elif risk_0_100 >= 40:
        level, recommendation = "elevated", "Investigate root cause; schedule retraining this cycle."
    elif risk_0_100 >= 20:
        level, recommendation = "watch", "Continue monitoring; no action required yet."
    else:
        level, recommendation = "healthy", "No action needed."

    return {
        "risk_score": risk_0_100,
        "level": level,
        "recommendation": recommendation,
        "components": {
            "feature_drift_pct": round(feature_drift_score * 100, 1),
            "prediction_drift_pct": round(prediction_drift_score * 100, 1),
            "concept_drift_pct": round(concept_drift_score * 100, 1),
        },
        "weights": WEIGHTS,
    }
