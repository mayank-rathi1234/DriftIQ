# 🛰️ Data Drift Observatory

An ML monitoring system that goes beyond "feature X has drifted" to answer
the question that actually matters in production:

> **Is this drift harmful to the model, why, and how much runway do we
> have before it fails?**

It detects feature drift, prediction drift, and concept drift separately,
attributes performance risk to specific features, forecasts near-term
accuracy decay with honest backtested validation, decides whether to
trigger a retrain, and includes a **benchmark experiment** that empirically
measures detector reliability across six distinct, simulated drift regimes.

## Why this exists (a research question, not just a dashboard)

> **Which drift-detection methods are most reliable under different types
> and magnitudes of distribution shift?**

Answered empirically in [`results/benchmark_summary.csv`](results/benchmark_summary.csv)
by simulating six ground-truth drift scenarios and scoring five statistical
detectors against a known onset point (AUROC, detection lag, false-positive
rate) — see [Benchmark results](#benchmark-results).

## Architecture

```
TRAINING DATA → Baseline Profile → Trained ML Model (XGBoost)
                                          │
                                   PRODUCTION DATA (6 simulated scenarios)
                                          │
                              ┌───────────┴───────────┐
                              │     Drift Detection    │
                              └───────────┬───────────┘
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             Feature Drift        Prediction Drift        Concept Drift
        (KS/PSI/KL/JS/EMD    (KS/PSI/JS/EMD on model    (domain classifier +
         per feature)          output distribution)     delayed-label corr.
                                                          + MI shift)
                    └─────────────────────┼─────────────────────┘
                                          ▼
                    Baseline-calibrated risk scoring (calibration.py)
                                          ▼
                              Root Cause Analysis
                        (drift severity × model feature importance)
                                          ▼
                     Performance Prediction + Backtested Validation
                                          ▼
                    🚨 Model Risk Score (0-100) → Retrain Decision
                                          ▼
              FastAPI service + Streamlit dashboard + MLflow logging
```

## This project was built, then deliberately audited and fixed

The first version of this project worked end-to-end but had five real bugs,
found by writing behavioral tests rather than just unit tests. They're left
documented here (and in the relevant module docstrings) because the fixes
are more informative than the fact that a bug existed:

| Bug | How it was found | Fix |
|---|---|---|
| Categorical features (`device_type`, `merchant_category`) always got zero importance in root-cause ranking | Manual inspection of ranked output — categoricals never appeared, ever, in any scenario | Root cause was a silent name mismatch (`device_type` vs the model's `device_type_enc` column); fixed via an explicit `RAW_TO_MODEL_COL` mapping in `train.py` |
| The "control" (no-drift) scenario scored ~65-90/100 risk | `test_control_scenario_stays_low_risk` failed | Min-max normalization stretches *any* range — including pure noise — to fill [0,1]. Replaced with burn-in-calibrated z-scores (`calibration.py`) that ask "is this far from established normal noise?" instead of "is this the batch-relative max?" |
| Concept-drift score saturated to ~1.0 from batch 5 onward in every scenario, including well before real drift | `test_concept_drift_dominates_risk_breakdown` failed after the calibration fix, exposing a second bug underneath the first | The MI-shift metric divided by its own (often tiny) baseline value, so pure estimation noise produced ratios near 1.0. Fixed by reporting raw MI difference and letting the calibration layer establish the noise floor, instead of pre-normalizing by an unstable denominator |
| Performance-forecast validation (`validation_mae`) was `None` most of the time | Code review — it only validated against batches that happened to already exist past the forecast horizon | Replaced with proper walk-forward backtesting: refit using only data available *h* batches before "today," forecast forward, compare against now-known truth. Always populated given enough history |
| Root-cause/drift-score normalization used the full simulated history, including future batches | Code review | Rewritten to use only batches ≤ "today" (causal/expanding), consistent with what a real production system could actually see |
| Risk score swung wildly batch-to-batch (e.g. 60 → 25 → 27 → 12) even while true drift was constant and sustained | Manual trend inspection across consecutive batches after the calibration fix above | Each sub-score was a single noisy point estimate (MI/domain-classifier on ~600 rows). Added EWMA smoothing (`calibration.smoothed_calibrated_score`, span=3) over the calibrated history, damping single-batch noise while staying responsive to sustained regime change |

Two of these (control-scenario false alarm, concept-score saturation) were
only caught because pipeline-level tests asserted *behavior* ("control
should stay low-risk," "concept drift should dominate its own scenario"),
not just that individual functions ran without crashing. A third
(batch-to-batch risk-score instability) was only caught by manually
tracing the same scenario across consecutive batches and noticing the
score had no business moving that much given the ground truth hadn't
changed. That's the argument for `tests/test_pipeline.py` and for
tracing real output over synthetic-but-plausible time series, rather than
trusting a single successful run — and it's a real example of exactly
that value being delivered, not a hypothetical one.

## What makes the concept-drift handling honest, not hand-waved

Concept drift (P(y\|X) changing) can't be measured the same way as feature
drift, because in production you rarely have real-time labels. This
project implements two real, distinct signals, each calibrated
independently before being combined:

1. **Domain classifier AUC** — a classifier trained to tell reference vs.
   current-window rows apart using only features. AUC ≈ 0.5 means inputs
   *look* unchanged (no covariate shift) — so if the model's behavior
   still shifts, that isolates concept drift from feature drift.
2. **Delayed-label relationship shift** — labels for batch *b* only
   "arrive" 5 batches later, mirroring real production lag. Once
   available, both a **linear** signal (Pearson correlation shift) and a
   **nonlinear** signal (mutual information shift) are computed, since
   correlation alone would miss a feature whose relationship to the label
   went from negative to U-shaped while staying near-zero correlation
   throughout. The two are calibrated against their own noise baselines
   separately, then combined via `max()` — never combined before
   calibration (see the bug table above for why that matters).

## Benchmark results

Six ground-truth-tagged scenarios were simulated (`n=8,000` training rows,
30 production batches each) and every statistical detector was scored
against the known drift onset batch:

| Scenario | Best detector | Mean AUROC | Worst detector | Mean AUROC |
|---|---|---|---|---|
| A. Customer population shift | JS divergence | 0.696 | Wasserstein | 0.572 |
| B. Feature distribution shift | KS statistic | 0.664 | PSI | 0.583 |
| C. True concept drift (relationship change) | KL divergence | 0.494 | KS statistic | 0.424 |
| D. Sudden outliers | KL divergence | 0.522 | Wasserstein | 0.476 |
| E. Gradual drift | Wasserstein | 0.733 | PSI | 0.637 |
| F. Seasonal/periodic drift | PSI | 0.530 | Wasserstein | 0.493 |

**The headline finding:** in Scenario C — genuine concept drift, where the
relationship between features and the label changes but the feature
*distributions themselves barely move* — every single feature-based
statistical test collapses to ~chance-level AUROC (0.42–0.49). This is the
empirical proof of why feature-drift monitoring alone is insufficient and
why this project builds a separate, independently-calibrated concept-drift
module.

Full per-feature results: [`results/benchmark_raw.csv`](results/benchmark_raw.csv) ·
Chart: [`results/benchmark_auroc.png`](results/benchmark_auroc.png)

Reproduce it:
```bash
PYTHONPATH=src python3 -m ddo.benchmark
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the benchmark experiment
PYTHONPATH=src python3 -m ddo.benchmark

# Launch the API
uvicorn ddo.api:app --app-dir src --reload --port 8000
# then POST http://localhost:8000/health-report  {"scenario": "C_relationship_change"}

# Launch the dashboard
streamlit run dashboard/app.py

# Run tests (includes the behavioral regression tests above)
pytest tests/

# Or run everything containerized
docker build -t drift-observatory .
docker run -p 8000:8000 drift-observatory
```

## Example output — Model Health Report (real, current run)

```json
{
  "scenario": "C_relationship_change",
  "as_of_batch": 22,
  "risk": {
    "risk_score": 30.9,
    "level": "watch",
    "components": {
      "feature_drift_pct": 4.7,
      "prediction_drift_pct": 22.0,
      "concept_drift_pct": 51.4
    }
  },
  "most_affected_features": [
    {"feature": "transaction_amount", "root_cause_score": 0.338},
    {"feature": "account_tenure_days", "root_cause_score": 0.251}
  ],
  "performance": {
    "current_accuracy": 0.8733,
    "forecast_by_batches_ahead": {"5": 0.8469, "10": 0.8153, "20": 0.752},
    "backtested_overall_mae": 0.0724
  },
  "retrain_decision": {
    "should_retrain": false,
    "urgency": "none",
    "reason": "Risk within acceptable range"
  }
}
```
This is real, computed output — note that `feature_drift_pct` (4.7%) is
low here while `concept_drift_pct` (51.4%) is the dominant signal, which
is exactly the case this project is designed to catch: features look
almost unchanged, but the relationship they have to the outcome hasn't.
`backtested_overall_mae` (0.072) is a genuine walk-forward validation
error, not an asserted confidence number.

## Tech stack

Python · Pandas / NumPy · SciPy (KS, Wasserstein) · Scikit-learn
(permutation importance, Ridge, domain classifier, mutual information) ·
XGBoost · FastAPI · Streamlit · MLflow · Docker · Matplotlib

## Project structure

```
src/ddo/
  scenarios.py            6 ground-truth drift scenario simulators
  detectors.py            KS, PSI, KL, JS, Wasserstein — implemented directly
  monitor.py              per-batch feature + prediction drift orchestration
  calibration.py          burn-in noise-calibrated z-score normalization
  concept_drift.py        domain classifier + correlation/MI relationship shift
  root_cause.py           severity × model-importance feature attribution
  performance_predictor.py  drift trend → forecast + walk-forward backtest
  risk_score.py           weighted 0-100 risk aggregation
  retrain_trigger.py      should_retrain() decision policy + notify() stub
  train.py                model training, encoding, raw↔model name mapping
  pipeline.py             end-to-end orchestration → Model Health Report
  benchmark.py            the research experiment
  api.py                  FastAPI service
  mlflow_logging.py       optional experiment tracking
dashboard/app.py          Streamlit UI
tests/
  test_detectors.py       statistical primitive sanity checks
  test_pipeline.py        behavioral regression tests (caught 2 real bugs)
results/                  benchmark CSVs + chart
Dockerfile
```

## What this demonstrates

- **Statistics**: KS test, PSI, KL divergence, Wasserstein distance,
  Jensen-Shannon divergence, mutual information — implemented from
  definitions, not just called from a library
- **ML**: gradient-boosted classification (XGBoost), regression forecasting
  (Ridge), permutation importance, domain-adversarial classifiers
- **ML engineering**: an experiment design with ground-truth labels used to
  validate detection methods empirically; walk-forward backtesting instead
  of untested forecasts; calibration against a noise baseline instead of
  naive normalization
- **Judgment**: found and fixed real bugs via behavioral testing (not just
  unit testing), and documented them rather than hiding them — including
  one bug that only became visible after fixing a different one

## Honest limitations (still true after the fixes above)

- The burn-in calibration window (first 5 batches) assumes those batches
  are drift-free. If real drift started at batch 0, the baseline itself
  would be contaminated — a real deployment should calibrate against a
  known-clean historical window, not just "whatever arrived first."
- EWMA smoothing (span=3) trades detection latency for stability: a
  sudden, severe drift will take a few batches to fully register in the
  smoothed score rather than spiking immediately. `D_sudden_outliers`
  demonstrates this directly — its risk score is already fading back down
  by batch 25 even though the outlier injection itself was real, because
  the effect was transient and the smoothing (correctly) doesn't treat a
  two-batch blip as a sustained regime change.
- Performance forecasting uses a simple Ridge model on drift trends; the
  backtested MAE is genuine, but the specific numbers are dataset-specific
  and would need recalibration per real deployment.
- The domain-classifier concept-drift signal is indirect; the
  correlation/MI-shift signal is direct but requires labels, which is the
  real-world constraint being modeled, not avoided.
- Scenarios are synthetic. The benchmark methodology (known onset,
  AUROC/lag/FPR scoring) generalizes to real labeled incident data if you
  have it — the six regimes here are a stand-in for that.
