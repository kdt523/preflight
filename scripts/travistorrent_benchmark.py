"""Benchmark on TravisTorrent with the same rolling-origin CV.

    uv run python scripts/travistorrent_benchmark.py --csv data/travistorrent_8_2_2017.csv \
        [--projects rails/rails apache/jackrabbit-oak]
"""

import argparse

from preflight_core.data.travistorrent import TT_FEATURES, load
from preflight_core.models.evaluate import compare, write_report
from preflight_core.models.zoo import MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--projects", nargs="*")
    ap.add_argument(
        "--models", nargs="+", default=["always_pass", "logistic_regression", "lgbm_calibrated"]
    )
    args = ap.parse_args()

    df = load(args.csv, args.projects)
    table, _ = compare({n: MODELS[n] for n in args.models if n in MODELS}, df, TT_FEATURES)
    path = write_report({"TravisTorrent (rolling-origin CV)": table})
    print(table.to_string(index=False))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
