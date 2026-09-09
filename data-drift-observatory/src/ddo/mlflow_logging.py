"""
mlflow_logging.py
------------------
Optional MLflow integration: logs each Model Health Report as an MLflow run
(local file-based tracking store by default, no server required) so a
history of risk scores, drift components, and retrain decisions across
monitoring runs is queryable via `mlflow ui` instead of only living inside
one process's return value.

This is intentionally opt-in (call `log_report` explicitly) rather than
wired automatically into every pipeline call, since not every use of this
project (e.g. the benchmark experiment, unit tests) needs or wants a
tracking run created.
"""

import mlflow


def log_report(report: dict, experiment_name: str = "data-drift-observatory"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{report['scenario']}_batch{report['as_of_batch']}"):
        mlflow.log_param("scenario", report["scenario"])
        mlflow.log_param("as_of_batch", report["as_of_batch"])

        mlflow.log_metric("risk_score", report["risk"]["risk_score"])
        mlflow.log_metric("feature_drift_pct", report["risk"]["components"]["feature_drift_pct"])
        mlflow.log_metric("prediction_drift_pct", report["risk"]["components"]["prediction_drift_pct"])
        mlflow.log_metric("concept_drift_pct", report["risk"]["components"]["concept_drift_pct"])

        if report["performance"]["current_accuracy"] is not None:
            mlflow.log_metric("current_accuracy", report["performance"]["current_accuracy"])
        for h, acc in report["performance"]["forecast_by_batches_ahead"].items():
            mlflow.log_metric(f"forecast_acc_plus{h}", acc)
        if report["performance"]["backtested_overall_mae"] is not None:
            mlflow.log_metric("backtested_overall_mae", report["performance"]["backtested_overall_mae"])

        mlflow.log_param("retrain_should_retrain", report["retrain_decision"]["should_retrain"])
        mlflow.log_param("retrain_urgency", report["retrain_decision"]["urgency"])

        return mlflow.active_run().info.run_id
