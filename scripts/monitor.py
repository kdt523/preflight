"""Drift + live-performance monitor. Scheduled hourly (EventBridge on AWS, cron locally).

    uv run python scripts/monitor.py [--auto-retrain]

Pushes gauges to the Prometheus Pushgateway (PUSHGATEWAY_URL) so Grafana alerts fire;
writes reports/monitor_latest.json; optionally kicks off retraining.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prometheus_client import REGISTRY, push_to_gateway

from preflight_core.db.engine import get_engine
from preflight_core.models.registry import BundleProvider, get_registry
from preflight_core.monitoring.monitor import run_monitor
from preflight_core.observability.metrics import RETRAIN_TRIGGERED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--auto-retrain", action="store_true")
    args = ap.parse_args()

    bundle = BundleProvider(get_registry(), reload_s=0).champion()
    if bundle is None:
        raise SystemExit("no champion model yet")
    report = run_monitor(get_engine(), bundle, args.window_days)
    for reason in report.retrain_reasons:
        RETRAIN_TRIGGERED.labels(reason.split(":")[0]).inc()
    if gw := os.environ.get("PUSHGATEWAY_URL"):
        push_to_gateway(gw, job="preflight-monitor", registry=REGISTRY)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/monitor_latest.json").write_text(
        json.dumps(report.as_dict(), indent=2, default=str)
    )
    print(json.dumps(report.as_dict(), indent=2, default=str))
    if args.auto_retrain and report.retrain_reasons:
        sys.exit(subprocess.call([sys.executable, "scripts/retrain.py"]))


if __name__ == "__main__":
    main()
