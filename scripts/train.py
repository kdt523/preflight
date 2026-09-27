"""Train a bundle from the feature dataset, save it, log it, register it as a candidate.

    uv run python scripts/train.py [--folds 5] [--per-repo-min 500]

Promotion (candidate -> shadow -> champion) is scripts/promote.py (week 5).
"""

import argparse
import json
import logging

from preflight_core.db.engine import get_engine
from preflight_core.features.dataset import load_dataset
from preflight_core.logging import setup_logging
from preflight_core.models.registry import LocalRegistry, get_registry
from preflight_core.models.test_ranker import load_test_runs
from preflight_core.models.tracking import log_to_mlflow, record_model_run
from preflight_core.models.train import train_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-repo-min", type=int, default=500)
    ap.add_argument("--cost-false-alarm", type=float, default=3.0)
    ap.add_argument("--cost-miss", type=float, default=1.0)
    ap.add_argument("--no-tests", action="store_true", help="skip the per-test ranker")
    args = ap.parse_args()
    setup_logging()
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    engine = get_engine()
    runs = [] if args.no_tests else load_test_runs(engine)
    bundle = train_bundle(
        load_dataset(),
        runs or None,
        n_folds=args.folds,
        per_repo_min=args.per_repo_min,
        c_fp=args.cost_false_alarm,
        c_fn=args.cost_miss,
    )
    path = bundle.save()
    run_id = log_to_mlflow(bundle, path)
    record_model_run(engine, bundle)
    reg = get_registry()
    if isinstance(reg, LocalRegistry):
        reg.add_candidate(bundle.version)
    else:  # MLflow: the model version points at this run's `bundle` artifact
        reg.add_candidate(bundle.version, run_id)
    print(
        json.dumps(
            {
                "version": bundle.version,
                "saved": str(path),
                "mlflow_run": run_id,
                "global_cv": bundle.metrics["global_cv"],
                "blame_free_cost": bundle.metrics["blame_free_pr_auc_cost"],
                "test_ranker": bundle.metrics["test_ranker"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
