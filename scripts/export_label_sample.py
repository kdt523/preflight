"""Export failed builds with log tails for hand-labelling (target ~300, stratified by label).

    uv run python scripts/export_label_sample.py --n 300 --out data/label_sample.csv

Fill the `human_label` column with one of: code_defect, flaky, infra, config.
Week 2 measures the labeler's precision per class/source against these.
"""

import argparse
import csv
import random
from pathlib import Path

from sqlalchemy import text

from preflight_core.db.engine import get_engine
from preflight_core.storage.s3 import get_raw_store

QUERY = text(
    """
    SELECT b.id AS build_id, r.full_name AS repo, b.run_id, b.run_attempt, b.failure_type,
           b.label_source, b.label_reason, j.id AS job_id, j.name AS job_name, j.failed_step,
           j.log_tail_key
    FROM builds b
    JOIN repos r ON r.id = b.repo_id
    JOIN jobs j ON j.build_id = b.id
    WHERE b.failure_type IN ('code_defect', 'flaky', 'infra', 'config')
      AND j.conclusion IN ('failure', 'timed_out') AND j.log_tail_key IS NOT NULL
    """
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/label_sample.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with get_engine().connect() as c:
        rows = [r._asdict() for r in c.execute(QUERY)]
    rng = random.Random(args.seed)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["failure_type"], []).append(r)
    per = max(1, args.n // max(1, len(by_label)))
    sample = [r for group in by_label.values() for r in rng.sample(group, min(per, len(group)))]
    rng.shuffle(sample)

    store = get_raw_store()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=[*sample[0].keys(), "log_tail", "human_label"] if sample else ["empty"]
        )
        w.writeheader()
        for r in sample:
            r["log_tail"] = "\n".join(
                store.get_bytes(r["log_tail_key"]).decode().splitlines()[-60:]
            )
            r["human_label"] = ""
            w.writerow(r)
    print(f"{len(sample)} rows -> {args.out}  (from {len(rows)} candidates, {per}/label max)")


if __name__ == "__main__":
    main()
