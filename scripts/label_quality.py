"""Measure labeler precision against hand labels, then train the log classifier.

    uv run python scripts/label_quality.py --csv data/label_sample.csv

Input: the CSV from export_label_sample.py with `human_label` filled in
(code_defect | flaky | infra | config). Rows left blank are ignored.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from preflight_core.labeling.log_classifier import LogClassifier, label_quality, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/label_sample.csv")
    ap.add_argument("--model-out", default="models/log_classifier.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).fillna({"human_label": "", "failed_step": ""})
    df = df[df["human_label"].str.strip() != ""]
    if len(df) < 20:
        raise SystemExit(f"only {len(df)} hand-labelled rows; label more first")
    report = label_quality(df)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/label_quality.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("n", "rule_coverage", "rule_accuracy_when_covered", "per_class")
            },
            indent=2,
        )
    )
    LogClassifier(train(df)).save(args.model_out)
    print(f"classifier -> {args.model_out}; full report -> reports/label_quality.json")


if __name__ == "__main__":
    main()
