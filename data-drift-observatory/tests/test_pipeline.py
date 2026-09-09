"""
Pipeline-level tests: check end-to-end BEHAVIOR, not just detector math.
These are slower (they train models + run the full monitoring pipeline)
so they're kept few and targeted at the claims the README actually makes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ddo.pipeline import prepare_system, generate_health_report
from ddo.retrain_trigger import should_retrain, RetrainDecision


@pytest.fixture(scope="module")
def control_report():
    system = prepare_system("control", n_train=4000, seed=1)
    report, _ = generate_health_report(system, as_of_batch=20)
    return report


@pytest.fixture(scope="module")
def concept_drift_report():
    system = prepare_system("C_relationship_change", n_train=4000, seed=1)
    report, _ = generate_health_report(system, as_of_batch=25)
    return report


def test_control_scenario_stays_low_risk(control_report):
    """The negative control (no injected drift) should never trigger a
    high risk score -- if it does, something upstream is miscalibrated."""
    assert control_report["risk"]["risk_score"] < 40
    assert control_report["risk"]["level"] in ("healthy", "watch")


def test_concept_drift_dominates_risk_breakdown(concept_drift_report):
    """For the true-concept-drift scenario, concept drift should be the
    largest component of the risk score -- this is the project's central
    empirical claim and should hold structurally, not just anecdotally."""
    comps = concept_drift_report["risk"]["components"]
    assert comps["concept_drift_pct"] >= comps["feature_drift_pct"]
    assert comps["concept_drift_pct"] >= comps["prediction_drift_pct"]


def test_root_cause_includes_nonzero_categorical_importance():
    """Regression test for the raw-name / model-column-name mismatch bug:
    categorical features must be able to receive nonzero importance_norm,
    not be silently zeroed by a naming mismatch."""
    system = prepare_system("A_customer_population", n_train=4000, seed=1)
    _, artifacts = generate_health_report(system, as_of_batch=20)
    rc = artifacts["root_cause_df"]
    cat_rows = rc[rc["feature"].isin(["device_type", "merchant_category"])]
    assert (cat_rows["importance_norm"] > 0).any()


def test_backtested_mae_is_populated(concept_drift_report):
    """The forecaster's validation MAE must actually compute a number,
    not silently return None -- regression test for the earlier
    'often null' validation bug."""
    perf = concept_drift_report["performance"]
    assert perf["backtested_overall_mae"] is not None
    assert perf["backtested_overall_mae"] >= 0


def test_retrain_decision_logic():
    healthy_risk = {"risk_score": 10.0, "components": {"concept_drift_pct": 5.0}}
    critical_risk = {"risk_score": 85.0, "components": {"concept_drift_pct": 20.0}}
    concept_override_risk = {"risk_score": 30.0, "components": {"concept_drift_pct": 75.0}}

    d1 = should_retrain(healthy_risk)
    assert isinstance(d1, RetrainDecision) and d1.should_retrain is False

    d2 = should_retrain(critical_risk)
    assert d2.should_retrain is True and d2.urgency == "immediate"

    d3 = should_retrain(concept_override_risk)
    assert d3.should_retrain is True and "Concept drift" in d3.reason
