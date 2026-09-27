"""Weekly Slack digest (SPEC feature 9). Every number is computed from the database now.

uv run python scripts/slack_digest.py --dry-run     # print the message
uv run python scripts/slack_digest.py               # post to SLACK_WEBHOOK_URL
"""

import argparse
import json
import math
from datetime import UTC, datetime, timedelta

import httpx

from apps.dashboard import data
from preflight_core.config import get_settings
from preflight_core.db.engine import get_engine


def pct(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def build_message(days: int = 7) -> dict:
    since = datetime.now(UTC) - timedelta(days=days)
    with get_engine().connect() as c:
        chk, out = data.checks(c, since), data.outcomes(c, since)
        repos = data.repos(c)
        hot = []
        for r in repos.itertuples():
            h = data.hotspots(c, r.id, days=30)
            if len(h):
                top = h.sort_values("fail_rate", ascending=False).iloc[0]
                hot.append(
                    f"• `{r.full_name}`: `{top.path}` ({pct(top.fail_rate)} smoothed "
                    f"fail rate, {int(top.builds)} builds)"
                )
    k = data.kpis(chk, out)
    bands = chk["band"].value_counts().to_dict() if len(chk) else {}
    lines = [
        f"*Preflight weekly* ({since:%b %d} – {datetime.now(UTC):%b %d})",
        f"{k['checks']} checks · 🟢 {bands.get('low', 0)} · 🟡 {bands.get('watch', 0)} · "
        f"🔴 {bands.get('high', 0)} · skipped {bands.get('skipped', 0)}",
        f"High-risk flags that really failed: *{pct(k.get('high_band_precision'))}* · "
        f"live PR-AUC {k.get('live_pr_auc', float('nan')):.3f} on {k['labelled']} known outcomes"
        if k.get("live_pr_auc") is not None
        else "Not enough known outcomes yet for accuracy.",
        f"Developers found checks useful: {pct(k['useful_share'])} ({k['votes']} votes)",
    ]
    if hot:
        lines += ["", "*Riskiest files (30 days)*", *hot[:5]]
    return {"text": "\n".join(lines)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    msg = build_message(args.days)
    url = get_settings().slack_webhook_url
    if args.dry_run or url is None:
        print(json.dumps(msg, indent=2, ensure_ascii=False))
        if url is None and not args.dry_run:
            print("(SLACK_WEBHOOK_URL not set: printed instead of posting)")
        return
    httpx.post(url.get_secret_value(), json=msg, timeout=10).raise_for_status()
    print("posted digest")


if __name__ == "__main__":
    main()
