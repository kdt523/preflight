"""Dashboard queries (SPEC feature 8). Plain functions over a Connection, so they run the same
under the pipeline role or a tenant-scoped (RLS) connection. No Streamlit imports here."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Connection, text

from preflight_core.features.dataset import unit_label
from preflight_core.features.state import FILE_PRIOR_STRENGTH, GLOBAL_PRIOR_RATE, smoothed
from preflight_core.models.metrics import ece, pr_auc, precision_recall_at, reliability


def repos(c: Connection) -> pd.DataFrame:
    return pd.DataFrame(
        c.execute(text("SELECT id, full_name, failure_base_rate FROM repos ORDER BY full_name"))
        .mappings()
        .all()
    )


def checks(c: Connection, since: datetime, repo_id: int | None = None) -> pd.DataFrame:
    """Champion predictions served (and fail-opens), one row each, with feedback."""
    rows = (
        c.execute(
            text("""
        SELECT p.id, p.repo_id, p.sha, p.band, p.p_fail, p.ci_plan, p.latency_ms,
               p.conformal_set, p.created_at, p.fail_open_reason,
               (SELECT bool_and(f.useful) FROM feedback f WHERE f.prediction_id = p.id) AS useful
        FROM predictions p
        WHERE NOT p.shadow AND p.created_at >= :since
          AND (CAST(:repo AS integer) IS NULL OR p.repo_id = :repo)
    """),
            {"since": since, "repo": repo_id},
        )
        .mappings()
        .all()
    )
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "repo_id",
            "sha",
            "band",
            "p_fail",
            "ci_plan",
            "latency_ms",
            "conformal_set",
            "created_at",
            "fail_open_reason",
            "useful",
        ],
    )


def outcomes(c: Connection, since: datetime, repo_id: int | None = None) -> pd.DataFrame:
    """Served predictions joined to the unit label CI later produced."""
    rows = (
        c.execute(
            text("""
        SELECT p.id, p.p_fail, p.band, p.created_at, array_agg(b.failure_type) AS types,
               EXTRACT(EPOCH FROM max(b.completed_at) - min(b.started_at)) / 60 AS minutes
        FROM predictions p
        JOIN builds b ON b.repo_id = p.repo_id AND b.sha = p.sha AND b.run_attempt = 1
        WHERE NOT p.shadow AND p.p_fail IS NOT NULL AND p.created_at >= :since
          AND (CAST(:repo AS integer) IS NULL OR p.repo_id = :repo)
        GROUP BY p.id
    """),
            {"since": since, "repo": repo_id},
        )
        .mappings()
        .all()
    )
    df = pd.DataFrame(rows, columns=["id", "p_fail", "band", "created_at", "types", "minutes"])
    if len(df):
        df["y"] = [unit_label(list(t)) for t in df["types"]]
        df = df[df["y"].notna()].astype({"y": int})
    return df


def kpis(chk: pd.DataFrame, out: pd.DataFrame) -> dict[str, Any]:
    served = chk[chk["band"] != "skipped"]
    votes = chk["useful"].dropna()
    k: dict[str, Any] = {
        "checks": len(chk),
        "fail_open_rate": (chk["band"] == "skipped").mean() if len(chk) else math.nan,
        "abstain_rate": served["conformal_set"].map(lambda s: s is not None and len(s) != 1).mean()
        if len(served)
        else math.nan,
        "p95_latency_ms": served["latency_ms"].quantile(0.95) if len(served) else math.nan,
        "useful_share": votes.mean() if len(votes) else math.nan,
        "votes": len(votes),
        "labelled": len(out),
    }
    if len(out) and out["y"].nunique() == 2:
        y, p = out["y"].to_numpy(), out["p_fail"].to_numpy()
        k["live_pr_auc"] = pr_auc(y, p)
        k["live_ece"] = ece(y, p)
        high = out["band"] == "high"
        k["high_band_precision"] = out.loc[high, "y"].mean() if high.any() else math.nan
        k["recall_at_high"] = precision_recall_at(y, (out["band"] == "high").to_numpy(), 0.5)[1]
    return k


def weekly(chk: pd.DataFrame, col: str = "id", how: str = "count") -> list[float]:
    """Weekly series for KPI sparklines (evenly spaced, oldest first)."""
    if not len(chk):
        return []
    s = chk.set_index(pd.to_datetime(chk["created_at"], utc=True))[col]
    return s.resample("W").agg(how).fillna(0).tolist()


def ci_minutes(out: pd.DataFrame, skippable_share: float) -> dict[str, float]:
    """ESTIMATE, not a measurement: minutes the lint+affected plan would skip on low-risk
    builds, assuming `skippable_share` of a full run is outside lint + affected tests."""
    low = out[out["band"] == "low"] if len(out) else out
    total = float(out["minutes"].sum()) if len(out) else 0.0
    skipped = float(low["minutes"].sum() * skippable_share) if len(low) else 0.0
    missed = int(low["y"].sum()) if len(low) else 0  # low-risk builds that did fail
    return {"total_minutes": total, "estimated_saved": skipped, "low_risk_failures": missed}


def hotspots(c: Connection, repo_id: int, days: int = 90) -> pd.DataFrame:
    """Per file: churn and smoothed failure rate over the window (the treemap)."""
    rows = (
        c.execute(
            text("""
        WITH first_builds AS (
          SELECT sha, min(started_at) AS started_at, array_agg(failure_type) AS types
          FROM builds WHERE repo_id = :repo AND run_attempt = 1 AND started_at >= :since
          GROUP BY sha
        )
        SELECT f.path, fb.sha, f.added + f.deleted AS churn, fb.types
        FROM first_builds fb JOIN file_changes f ON f.repo_id = :repo AND f.sha = fb.sha
    """),
            {"repo": repo_id, "since": datetime.now(UTC) - timedelta(days=days)},
        )
        .mappings()
        .all()
    )
    df = pd.DataFrame(rows, columns=["path", "sha", "churn", "types"])
    if not len(df):
        return pd.DataFrame(columns=["path", "dir", "churn", "builds", "fails", "fail_rate"])
    df["y"] = [unit_label(list(t)) for t in df["types"]]
    base = df.drop_duplicates("sha")["y"].dropna()
    prior = float(base.mean()) if len(base) else GLOBAL_PRIOR_RATE
    g = (
        df.groupby("path")
        .agg(churn=("churn", "sum"), builds=("y", "count"), fails=("y", "sum"))
        .reset_index()
    )
    g["fail_rate"] = [
        smoothed(f, b, prior, FILE_PRIOR_STRENGTH)
        for f, b in zip(g["fails"], g["builds"], strict=True)
    ]
    g["dir"] = g["path"].map(lambda p: p.rsplit("/", 1)[0] if "/" in p else "(root)")
    return g.sort_values("churn", ascending=False)


def calibration(out: pd.DataFrame) -> pd.DataFrame:
    if not len(out):
        return pd.DataFrame(columns=["bin", "n", "predicted", "observed"])
    return pd.DataFrame(reliability(out["y"].to_numpy(), out["p_fail"].to_numpy()))


def monitor_report(path: str | Path = "reports/monitor_latest.json") -> dict[str, Any] | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
