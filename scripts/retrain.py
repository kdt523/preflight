"""Retrain end-to-end and start the new bundle in shadow mode.

    uv run python scripts/retrain.py

Refreshes labels + features from the latest data, trains, registers a candidate, and puts
it in shadow if its CV PR-AUC is at least the champion's. Promotion happens only after the
7-day live comparison (scripts/promote.py evaluate / promote).
"""

import json
import subprocess
import sys

from preflight_core.models.bundle import PreflightBundle
from preflight_core.models.registry import get_registry


def run(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call([sys.executable, *cmd])


if __name__ == "__main__":
    run("scripts/label.py")
    run("scripts/build_features.py")
    out = subprocess.check_output([sys.executable, "scripts/train.py"], text=True)
    print(out)
    new = json.loads(out[out.index("{") :])
    reg = get_registry()
    champ = reg.read().get("champion")
    new_auc = new["global_cv"].get("pr_auc_mean") or 0
    old_auc = 0.0
    if champ:
        old = PreflightBundle.load(reg.bundle_path(champ))
        old_auc = old.metrics["global_cv"].get("pr_auc_mean") or 0
    if new_auc >= old_auc:
        run("scripts/promote.py", "shadow", new["version"])
        if not champ:
            run("scripts/promote.py", "promote", new["version"], "--force")
    else:
        print(
            f"candidate {new['version']} CV PR-AUC {new_auc:.3f} < champion {old_auc:.3f}; "
            "not shadowed"
        )
