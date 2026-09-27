"""Watch loop (SPEC §2): feature drift (PSI) + live PR-AUC -> metrics, alerts, retrain triggers."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from preflight_core.models.bundle import PreflightBundle
from preflight_core.models.promotion import LiveMetrics, labelled_predictions
from preflight_core.monitoring.drift import PSI_ALERT, psi_max, psi_table
from preflight_core.observability.metrics import DRIFT_PSI, DRIFT_PSI_MAX, ROLLING_PR_AUC

MIN_SERVED_FOR_DRIFT = 200
MIN_POSITIVES_FOR_PERF = 30
PERF_DROP_RATIO = 0.85  # live PR-AUC below 85% of the CV estimate -> retrain
RETRAIN_EVERY = timedelta(days=7)


def served_features(engine: Engine, since: datetime) -> pd.DataFrame:
    with engine.connect() as c:
        rows = (
            c.execute(
                text(
                    "SELECT features FROM predictions WHERE created_at >= :since AND NOT shadow "
                    "AND features IS NOT NULL"
                ),
                {"since": since},
            )
            .scalars()
            .all()
        )
    return pd.DataFrame(list(rows)).astype(float) if rows else pd.DataFrame()


@dataclass
class MonitorReport:
    served: int
    psi: pd.DataFrame
    psi_max: float
    live: LiveMetrics
    cv_pr_auc: float | None
    retrain_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "served": self.served,
            "psi_max": self.psi_max,
            "psi_top": self.psi.head(10).to_dict(orient="records"),
            "live": {**self.live.__dict__, "first_seen": str(self.live.first_seen)},
            "cv_pr_auc": self.cv_pr_auc,
            "retrain_reasons": self.retrain_reasons,
        }


def retrain_reasons(
    served: int,
    pmax: float,
    live: LiveMetrics,
    cv_pr_auc: float | None,
    trained_at: datetime,
    now: datetime,
) -> list[str]:
    reasons = []
    if served >= MIN_SERVED_FOR_DRIFT and pmax > PSI_ALERT:
        reasons.append(f"drift: max PSI {pmax:.2f} > {PSI_ALERT}")
    if (
        cv_pr_auc
        and live.positives >= MIN_POSITIVES_FOR_PERF
        and live.pr_auc == live.pr_auc
        and live.pr_auc < PERF_DROP_RATIO * cv_pr_auc
    ):
        reasons.append(
            f"performance: live PR-AUC {live.pr_auc:.3f} < "
            f"{PERF_DROP_RATIO:.0%} of CV {cv_pr_auc:.3f}"
        )
    if now - trained_at >= RETRAIN_EVERY:
        reasons.append("schedule: champion older than 7 days")
    return reasons


def run_monitor(
    engine: Engine, bundle: PreflightBundle, window_days: int = 7, now: datetime | None = None
) -> MonitorReport:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=window_days)
    X = served_features(engine, since)
    table = (
        psi_table(bundle.reference, X)
        if len(X)
        else pd.DataFrame(columns=["feature", "psi", "status"])
    )
    pmax = psi_max(table)
    live_df = labelled_predictions(engine, since, bundle.version)
    live = LiveMetrics.of(live_df[~live_df["shadow"]] if len(live_df) else live_df)
    cv = bundle.metrics.get("global_cv", {}).get("pr_auc_mean")

    DRIFT_PSI_MAX.set(pmax)
    for r in table.itertuples():
        DRIFT_PSI.labels(r.feature).set(r.psi)
    if live.pr_auc == live.pr_auc:
        ROLLING_PR_AUC.set(live.pr_auc)

    trained = datetime.fromisoformat(bundle.created_at)
    return MonitorReport(
        len(X), table, pmax, live, cv, retrain_reasons(len(X), pmax, live, cv, trained, now)
    )
