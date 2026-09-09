"""
dashboard/app.py
-----------------
Streamlit dashboard: pick a scenario, pick "today" (batch), see the live
Model Health Report plus the underlying drift-metric time series and the
detector benchmark leaderboard.

Run:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st

from ddo.pipeline import prepare_system, generate_health_report
from ddo.scenarios import SCENARIOS

st.set_page_config(page_title="Data Drift Observatory", page_icon="🛰️", layout="wide")
st.title("🛰️ Data Drift Observatory")
st.caption("Detects data/concept drift, diagnoses root cause, and predicts model degradation.")

col1, col2 = st.columns([2, 1])
with col1:
    scenario = st.selectbox(
        "Production scenario",
        options=list(SCENARIOS.keys()),
        format_func=lambda k: f"{k} — {SCENARIOS[k].description}",
        index=list(SCENARIOS.keys()).index("C_relationship_change"),
    )
with col2:
    n_batches = SCENARIOS[scenario].n_batches
    as_of_batch = st.slider("As-of batch (simulated day)", 5, n_batches - 1, n_batches - 1)

if "system" not in st.session_state or st.session_state.get("scenario") != scenario:
    with st.spinner("Training model + simulating production stream..."):
        st.session_state["system"] = prepare_system(scenario, n_train=8000)
        st.session_state["scenario"] = scenario

system = st.session_state["system"]

with st.spinner("Running drift monitoring pipeline..."):
    report, artifacts = generate_health_report(system, as_of_batch=as_of_batch)

st.header("Model Health Report")
risk = report["risk"]
level_color = {"critical": "🔴", "elevated": "🟠", "watch": "🟡", "healthy": "🟢"}[risk["level"]]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Risk", f"{level_color} {risk['risk_score']}/100")
c2.metric("Feature drift", f"{risk['components']['feature_drift_pct']}%")
c3.metric("Prediction drift", f"{risk['components']['prediction_drift_pct']}%")
c4.metric("Concept drift", f"{risk['components']['concept_drift_pct']}%")

st.info(f"**Recommendation:** {report['recommendation']}")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Most affected features")
    st.dataframe(pd.DataFrame(report["most_affected_features"]), hide_index=True)

with col_b:
    st.subheader("Estimated accuracy trajectory")
    perf = report["performance"]
    perf_df = pd.DataFrame(
        {"batches_ahead": [0] + list(perf["forecast_by_batches_ahead"].keys()),
         "accuracy": [perf["current_accuracy"]] + list(perf["forecast_by_batches_ahead"].values())}
    )
    st.line_chart(perf_df.set_index("batches_ahead"))

st.subheader("Drift metric trends over time")
fd = artifacts["feature_drift_df"]
metric = st.selectbox("Metric", ["psi", "ks", "js", "kl", "wasserstein"], index=0)
pivot = fd.pivot_table(index="batch", columns="feature", values=metric)
st.line_chart(pivot)
st.caption(f"Drift onset for this scenario is batch {SCENARIOS[scenario].onset_batch} (ground truth).")

st.subheader("Concept drift signals (domain classifier AUC, correlation shift, MI shift)")
st.line_chart(artifacts["concept_df"].set_index("batch")[
    ["domain_auc", "correlation_shift_score", "mi_shift_score"]
])
st.caption(
    "domain_auc ≈ 0.5 means features alone can't tell reference vs. current apart "
    "(no covariate shift). correlation_shift_score catches linear relationship changes; "
    "mi_shift_score catches nonlinear ones correlation would miss. Both are calibrated "
    "against their own noise baseline (not shown raw) before being combined into the "
    "risk score's concept_drift component."
)

with st.expander("Raw Model Health Report (JSON)"):
    st.json(report)
