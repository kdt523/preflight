"""Shadow -> champion promotion, decided on live outcomes rather than offline CV.

A shadow bundle scores every request (hidden). After at least MIN_SHADOW_DAYS we join both
models' predictions to the labels CI later produced and promote only if the shadow is at
least as good on PR-AUC and not meaningfully worse on calibration.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from preflight_core.features.dataset import unit_label
from preflight_core.models.metrics import ece, pr_auc

MIN_SHADOW_DAYS = 7
MIN_POSITIVES = 20
PR_AUC_TOLERANCE = 0.0
ECE_TOLERANCE = 0.02

_LABELLED = text(
    """
    SELECT p.model_version, p.shadow, p.repo_id, p.sha, p.p_fail, p.created_at,
           array_agg(b.failure_type) AS failure_types
    FROM predictions p
    JOIN builds b ON b.repo_id = p.repo_id AND b.sha = p.sha AND b.run_attempt = 1
    WHERE p.created_at >= :since AND p.p_fail IS NOT NULL
      AND (CAST(:version AS text) IS NULL OR p.model_version = :version)
    GROUP BY p.id
    """
)


def labelled_predictions(
    engine: Engine, since: datetime, version: str | None = None
) -> pd.DataFrame:
    """Served predictions joined to the unit label CI eventually produced."""
    with engine.connect() as c:
        rows = c.execute(_LABELLED, {"since": since, "version": version}).all()
    df = pd.DataFrame(
        [
            {
                "model_version": r.model_version,
                "shadow": r.shadow,
                "repo_id": r.repo_id,
                "sha": r.sha,
                "p": r.p_fail,
                "created_at": r.created_at,
                "y": unit_label(list(r.failure_types)),
            }
            for r in rows
        ],
        columns=["model_version", "shadow", "repo_id", "sha", "p", "created_at", "y"],
    )
    return df[df["y"].notna()].astype({"y": int}) if len(df) else df


@dataclass(frozen=True)
class LiveMetrics:
    n: int
    positives: int
    pr_auc: float
    ece: float
    first_seen: datetime | None

    @classmethod
    def of(cls, df: pd.DataFrame) -> "LiveMetrics":
        if not len(df):
            return cls(0, 0, float("nan"), float("nan"), None)
        y, p = df["y"].to_numpy(), df["p"].to_numpy()
        both = len(set(y)) == 2
        return cls(
            len(df),
            int(y.sum()),
            pr_auc(y, p) if both else float("nan"),
            ece(y, p),
            df["created_at"].min(),
        )


def decide(
    champion: LiveMetrics, shadow: LiveMetrics, now: datetime | None = None
) -> tuple[bool, str]:
    now = now or datetime.now(UTC)
    if shadow.first_seen is None:
        return False, "shadow has no labelled predictions yet"
    if now - shadow.first_seen < timedelta(days=MIN_SHADOW_DAYS):
        return False, f"shadow has run < {MIN_SHADOW_DAYS} days"
    if shadow.positives < MIN_POSITIVES:
        return False, f"only {shadow.positives} failing builds observed (< {MIN_POSITIVES})"
    if np.isnan(champion.pr_auc):
        return True, "no comparable champion metrics; shadow meets minimums"
    if shadow.pr_auc < champion.pr_auc - PR_AUC_TOLERANCE:
        return False, f"shadow PR-AUC {shadow.pr_auc:.3f} < champion {champion.pr_auc:.3f}"
    if shadow.ece > champion.ece + ECE_TOLERANCE:
        return False, f"shadow ECE {shadow.ece:.3f} worse than champion {champion.ece:.3f}"
    return True, (
        f"shadow PR-AUC {shadow.pr_auc:.3f} >= champion {champion.pr_auc:.3f}, ECE {shadow.ece:.3f}"
    )


def compare_live(engine: Engine, champion: str, shadow: str, days: int = 14) -> dict[str, Any]:
    """Head-to-head on the SAME builds: only shas both models scored."""
    since = datetime.now(UTC) - timedelta(days=days)
    df = labelled_predictions(engine, since)
    a = df[(df["model_version"] == champion) & ~df["shadow"]]
    b = df[(df["model_version"] == shadow) & df["shadow"]]
    common = set(zip(a["repo_id"], a["sha"], strict=True)) & set(
        zip(b["repo_id"], b["sha"], strict=True)
    )
    key = lambda d: [(r, s) in common for r, s in zip(d["repo_id"], d["sha"], strict=True)]  # noqa: E731
    ca, cb = LiveMetrics.of(a[key(a)]), LiveMetrics.of(b[key(b)])
    ok, why = decide(ca, cb)
    return {"champion": ca.__dict__, "shadow": cb.__dict__, "promote": ok, "reason": why}
