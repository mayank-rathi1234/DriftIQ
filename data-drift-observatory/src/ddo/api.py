"""
api.py
------
Minimal FastAPI service wrapping the pipeline. In-memory model cache keyed
by scenario name (fine for a demo/portfolio project; a real deployment
would persist the trained model + baseline profile via MLflow/pickle and
load them here instead of retraining per process).

Run:  uvicorn ddo.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import prepare_system, generate_health_report
from .scenarios import SCENARIOS

app = FastAPI(
    title="Data Drift Observatory",
    description="Detects data/concept drift, diagnoses root cause, and forecasts model degradation.",
    version="0.1.0",
)

_SYSTEM_CACHE = {}


class HealthReportRequest(BaseModel):
    scenario: str = "C_relationship_change"
    as_of_batch: int | None = None


@app.get("/scenarios")
def list_scenarios():
    return {name: cfg.description for name, cfg in SCENARIOS.items()}


@app.post("/health-report")
def health_report(req: HealthReportRequest):
    if req.scenario not in SCENARIOS:
        raise HTTPException(400, f"Unknown scenario '{req.scenario}'. See /scenarios.")
    if req.scenario not in _SYSTEM_CACHE:
        _SYSTEM_CACHE[req.scenario] = prepare_system(req.scenario, n_train=8000)
    system = _SYSTEM_CACHE[req.scenario]
    report, _ = generate_health_report(system, as_of_batch=req.as_of_batch)
    return report


@app.get("/")
def root():
    return {
        "service": "Data Drift Observatory",
        "endpoints": ["/scenarios", "/health-report (POST)", "/docs"],
    }
