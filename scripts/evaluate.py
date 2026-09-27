"""Rolling-origin evaluation of baselines and models, with optional family ablations.

uv run python scripts/evaluate.py                         # baselines (week 2 exit)
uv run python scripts/evaluate.py --models all --ablate lgbm_calibrated
"""

import argparse

from preflight_core.features.dataset import load_dataset
from preflight_core.models.evaluate import (
    ablate_families,
    compare,
    per_repo,
    reliability_table,
    write_report,
)
from preflight_core.models.zoo import BASELINES, MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=BASELINES, help="names or 'all'")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ablate", help="model to run feature-family ablations for")
    args = ap.parse_args()

    names = list(MODELS) if args.models == ["all"] else args.models
    df = load_dataset()
    table, results = compare({n: MODELS[n] for n in names}, df, n_folds=args.folds)
    sections = {"models (rolling-origin CV)": table}
    best = max(results.values(), key=lambda r: r.summary().get("pr_auc_mean") or 0)
    if best.oof is not None:
        sections[f"per repo ({best.name})"] = per_repo(best.oof)
        sections[f"reliability ({best.name})"] = reliability_table(best.oof)
    if args.ablate:
        sections[f"ablation ({args.ablate})"] = ablate_families(MODELS[args.ablate], df, args.folds)
    path = write_report(sections)
    print(table.to_string(index=False))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
