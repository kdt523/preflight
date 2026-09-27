"""Experiment tracking: MLflow (when MLFLOW_TRACKING_URI is set) + the model_runs table."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import insert

from preflight_core.db.schema import model_runs
from preflight_core.models.bundle import PreflightBundle

log = logging.getLogger(__name__)
EXPERIMENT = "preflight"


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flat(v, f"{key}.")
        elif isinstance(v, int | float) and v == v:  # drop NaN
            out[key] = float(v)
    return out


def log_to_mlflow(bundle: PreflightBundle, bundle_dir: Path) -> str | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=bundle.version) as run:
        m = bundle.metrics
        mlflow.set_tags(
            {
                "version": bundle.version,
                "feature_set_version": bundle.feature_set_version,
                "feature_hash": bundle.feature_hash,
                "data_window": " .. ".join(bundle.data_window),
            }
        )
        mlflow.log_params(
            {
                "n_labelled": m["n_labelled"],
                "cost_false_alarm": m["costs"]["false_alarm"],
                "cost_miss": m["costs"]["miss"],
                "conformal_confidence": m["conformal_confidence"],
                "per_repo_models": ",".join(sorted(bundle.per_repo)) or "none",
            }
        )
        mlflow.log_metrics(
            {
                k.replace("@", "_"): v
                for k, v in _flat(
                    {
                        "global": m["global_cv"],
                        "blame_free": m["blame_free_cv"],
                        "test_ranker": m.get("test_ranker", {}),
                        "blame_free_pr_auc_cost": m["blame_free_pr_auc_cost"],
                    }
                ).items()
            }
        )
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        return run.info.run_id


def record_model_run(engine: Engine, bundle: PreflightBundle, status: str = "candidate") -> None:
    g = bundle.metrics["global_cv"]
    stmt = insert(model_runs).values(
        version=bundle.version,
        data_window=" .. ".join(bundle.data_window),
        feature_hash=bundle.feature_hash,
        pr_auc=g.get("pr_auc_mean"),
        ece=g.get("pooled_ece"),
        threshold=bundle.global_model.bands.high,
        status=status,
        metrics=json.loads(json.dumps(bundle.metrics, default=str)),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["version"], set_={"status": stmt.excluded.status}
    )
    with engine.begin() as c:
        c.execute(stmt)
