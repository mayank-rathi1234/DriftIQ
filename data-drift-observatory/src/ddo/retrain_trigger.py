"""
retrain_trigger.py
-------------------
Turns the risk score into an actual decision + action, not just a
recommendation string. `should_retrain` is a pure function (easy to unit
test); `notify` is a pluggable side-effecting call (webhook/Slack/email in
a real deployment) that's mocked/logged here rather than making a real
network call, since a resume project shouldn't silently POST to
localhost or someone's real webhook URL by default.
"""

from dataclasses import dataclass


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    urgency: str  # "immediate" | "scheduled" | "none"


def should_retrain(risk: dict, min_risk_for_retrain: float = 70.0,
                    concept_drift_override: float = 60.0) -> RetrainDecision:
    """
    Decision policy (documented, adjustable):
      - risk_score >= min_risk_for_retrain (default 70/100)  -> immediate retrain
      - OR concept_drift_pct alone >= concept_drift_override (default 60%)
        even if overall risk is lower -- concept drift is weighted highest
        in risk_score already, but this override exists because concept
        drift can be masked in the blended score if feature/prediction
        drift happen to be low that same batch.
    """
    score = risk["risk_score"]
    concept_pct = risk["components"]["concept_drift_pct"]

    if score >= min_risk_for_retrain:
        return RetrainDecision(True, f"Overall risk score {score} >= threshold {min_risk_for_retrain}", "immediate")
    if concept_pct >= concept_drift_override:
        return RetrainDecision(True, f"Concept drift {concept_pct}% >= override threshold {concept_drift_override}%", "immediate")
    if score >= 40:
        return RetrainDecision(True, f"Risk score {score} elevated; retrain on next scheduled cycle", "scheduled")
    return RetrainDecision(False, "Risk within acceptable range", "none")


def notify(decision: RetrainDecision, scenario: str, webhook_url: str = None) -> dict:
    """Stub notification. In production this would POST to Slack/PagerDuty/
    an MLOps orchestrator (e.g. trigger an Airflow DAG or MLflow retraining
    job). Here it just returns what WOULD be sent, so the trigger logic is
    demonstrably wired up without making an unsolicited network call."""
    payload = {
        "scenario": scenario,
        "should_retrain": decision.should_retrain,
        "urgency": decision.urgency,
        "reason": decision.reason,
    }
    return {"sent": False, "would_send_to": webhook_url or "(no webhook configured)", "payload": payload}
