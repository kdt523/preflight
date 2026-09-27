"""Generates dashboards/preflight.json. Re-run after editing: python build_dashboard.py

Colours: risk bands are *state*, so they use the validated status palette (good / warning /
critical) and are always labelled; other series use categorical slots in fixed order.
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
LAT = "sum by (le) (rate(preflight_score_latency_seconds_bucket[5m]))"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"

panels: list[dict] = []


def panel(
    title,
    targets,
    kind="timeseries",
    w=12,
    h=8,
    unit=None,
    overrides=(),
    desc="",
    thresholds=None,
    fixed=None,
):
    x = (len(panels) * w) % 24 if w < 24 else 0
    y = sum(p["gridPos"]["h"] for p in panels) // 2 if w < 24 else 100 + len(panels)
    defaults: dict = {"custom": {"lineWidth": 2, "fillOpacity": 0, "showPoints": "never"}}
    if unit:
        defaults["unit"] = unit
    if fixed:
        defaults["color"] = {"mode": "fixed", "fixedColor": fixed}
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["custom"]["thresholdsStyle"] = {"mode": "dashed"}
    panels.append(
        {
            "id": len(panels) + 1,
            "type": kind,
            "title": title,
            "description": desc,
            "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [
                {"refId": chr(65 + i), "expr": e, "legendFormat": lf, "datasource": DS}
                for i, (e, lf) in enumerate(targets)
            ],
            "fieldConfig": {
                "defaults": defaults,
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": n},
                        "properties": [
                            {"id": "color", "value": {"mode": "fixed", "fixedColor": c}}
                        ],
                    }
                    for n, c in overrides
                ],
            },
            "options": {
                "legend": {"displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }
    )


panel(
    "Scorer latency (p50 / p95)",
    [
        (
            f"histogram_quantile(0.5, {LAT})",
            "p50",
        ),
        (
            f"histogram_quantile(0.95, {LAT})",
            "p95",
        ),
    ],
    unit="s",
    overrides=[("p50", BLUE), ("p95", ORANGE)],
    thresholds=[{"color": "transparent", "value": None}, {"color": CRIT, "value": 0.3}],
    desc="SLO: p95 < 300 ms (dashed line)",
)
panel(
    "Checks by risk band",
    [
        ('sum(rate(preflight_predictions_total{band="low"}[15m])) * 3600', "low"),
        ('sum(rate(preflight_predictions_total{band="watch"}[15m])) * 3600', "watch"),
        ('sum(rate(preflight_predictions_total{band="high"}[15m])) * 3600', "high"),
        ("sum(rate(preflight_fail_open_total[15m])) * 3600", "skipped (fail-open)"),
    ],
    unit="short",
    overrides=[("low", GOOD), ("watch", WARN), ("high", CRIT), ("skipped (fail-open)", "#8a8987")],
    desc="Checks per hour. Bands use status colours and are always labelled.",
)
panel(
    "Feature drift: max PSI",
    [("preflight_drift_psi_max", "max PSI")],
    fixed=BLUE,
    thresholds=[{"color": "transparent", "value": None}, {"color": CRIT, "value": 0.25}],
    desc="PSI > 0.25 = significant shift (alert + retrain trigger)",
)
panel(
    "Live PR-AUC (served predictions, 7 d)",
    [("preflight_rolling_pr_auc", "live PR-AUC")],
    fixed=BLUE,
    unit="percentunit",
)
panel(
    "Webhook queue",
    [
        ("preflight_queue_lag_messages", "undelivered"),
        ("preflight_queue_pending_messages", "unacked"),
    ],
    overrides=[("undelivered", BLUE), ("unacked", ORANGE)],
)
panel(
    "Dead-lettered events (1 h)", [("sum(increase(preflight_dlq_total[1h]))", "DLQ")], fixed=ORANGE
)
panel(
    "Top drifting features",
    [("topk(8, preflight_drift_psi)", "{{feature}}")],
    kind="bargauge",
    w=24,
    h=8,
    thresholds=[
        {"color": BLUE, "value": None},
        {"color": WARN, "value": 0.1},
        {"color": CRIT, "value": 0.25},
    ],
)

dash = {
    "uid": "preflight",
    "title": "Preflight",
    "schemaVersion": 39,
    "version": 1,
    "time": {"from": "now-24h", "to": "now"},
    "refresh": "30s",
    "tags": ["preflight"],
    "panels": panels,
}
Path(__file__).with_name("dashboards").joinpath("preflight.json").write_text(
    json.dumps(dash, indent=2)
)
print(f"{len(panels)} panels")
