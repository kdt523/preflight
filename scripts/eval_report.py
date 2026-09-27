"""Week-3 exit: full evaluation report with ablations, reliability curve and PR curves.

    uv run python scripts/eval_report.py
Writes reports/eval_<ts>.md (+ .json) and reports/figures/*.png. Numbers come only from
this run on the stored dataset.
"""

import argparse
import logging
from pathlib import Path

from preflight_core.db.engine import get_engine
from preflight_core.features.dataset import load_dataset
from preflight_core.features.definitions import AUTHOR, feature_names
from preflight_core.models.evaluate import (
    ablate_families,
    compare,
    cross_validate,
    per_repo,
    reliability_table,
    write_report,
)
from preflight_core.models.plots import pr_curves, reliability_plot
from preflight_core.models.test_ranker import load_test_runs
from preflight_core.models.train import train_test_ranker
from preflight_core.models.zoo import MODELS

MAIN = ["always_pass", "files_changed_gt_n", "logistic_regression", "lgbm_calibrated"]
DECISIONS = [
    "lgbm_calibrated",
    "lgbm_smote",
    "lgbm_no_monotone",
    "lgbm_full_class_weight",
    "lgbm_no_class_weight",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    df = load_dataset()
    figs = Path("reports/figures")
    figs.mkdir(parents=True, exist_ok=True)

    table, res = compare({n: MODELS[n] for n in MAIN}, df, n_folds=args.folds)
    decisions, _ = compare({n: MODELS[n] for n in DECISIONS}, df, n_folds=args.folds)
    lgbm = res["lgbm_calibrated"]
    bf = cross_validate(
        "blame-free", MODELS["lgbm_calibrated"], df, feature_names({AUTHOR}), args.folds
    ).summary()
    sections = {
        "Models vs baselines": table,
        "Modelling-decision ablations": decisions,
        "Feature-family ablations (lgbm_calibrated)": ablate_families(
            MODELS["lgbm_calibrated"], df, args.folds
        ),
        "Blame-free mode (author features off)": {
            "pr_auc_with_author": lgbm.summary().get("pr_auc_mean"),
            "pr_auc_without_author": bf.get("pr_auc_mean"),
        },
    }
    if lgbm.oof is not None:
        sections["Per repo (lgbm_calibrated)"] = per_repo(lgbm.oof)
        sections["Reliability table (lgbm_calibrated)"] = reliability_table(lgbm.oof)
        reliability_plot(lgbm.oof, figs / "reliability.png")
        pr_curves(
            {n: r.oof for n, r in res.items() if r.oof is not None and n != "always_pass"},
            figs / "pr_curves.png",
        )
        sections["Figures"] = {
            "reliability": "figures/reliability.png",
            "pr_curves": "figures/pr_curves.png",
        }
    runs = load_test_runs(get_engine())
    if runs:
        sections["Per-test ordering (held-out newest 20%)"] = train_test_ranker(runs)[1]
    path = write_report(sections)
    print(table.to_string(index=False))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
