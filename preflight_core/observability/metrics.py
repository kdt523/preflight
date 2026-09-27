"""Prometheus metrics (SPEC §8: latency, queue lag, fail-open count, prediction distribution)."""

from prometheus_client import Counter, Gauge, Histogram

SCORE_LATENCY = Histogram(
    "preflight_score_latency_seconds",
    "Scorer latency per request (features + model + SHAP)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)
PREDICTIONS = Counter("preflight_predictions_total", "Predictions served", ["band", "model_key"])
P_FAIL = Histogram(
    "preflight_p_fail", "Distribution of served P(fail)", buckets=[i / 20 for i in range(1, 21)]
)
ABSTAIN = Counter("preflight_abstentions_total", "Conformal abstentions")
SCORE_ERRORS = Counter("preflight_score_errors_total", "Scorer errors", ["kind"])

WEBHOOKS = Counter("preflight_webhooks_total", "Webhooks received", ["event", "status"])
FAIL_OPEN = Counter("preflight_fail_open_total", "Checks skipped by fail-open", ["reason"])
QUEUE_LAG = Gauge("preflight_queue_lag_messages", "Stream entries not yet delivered to workers")
QUEUE_PENDING = Gauge("preflight_queue_pending_messages", "Delivered but unacknowledged")
DLQ = Counter("preflight_dlq_total", "Messages moved to the dead-letter stream", ["event"])
EVENTS = Counter(
    "preflight_events_processed_total", "Worker events processed", ["event", "outcome"]
)

DRIFT_PSI_MAX = Gauge("preflight_drift_psi_max", "Max PSI across features (served vs training)")
DRIFT_PSI = Gauge("preflight_drift_psi", "PSI per feature", ["feature"])
ROLLING_PR_AUC = Gauge("preflight_rolling_pr_auc", "PR-AUC of served predictions, last window")
RETRAIN_TRIGGERED = Counter("preflight_retrain_triggered_total", "Retrains triggered", ["reason"])
