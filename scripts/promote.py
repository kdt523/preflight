"""Model lifecycle: candidate -> shadow -> champion, with one-command rollback.

uv run python scripts/promote.py status
uv run python scripts/promote.py shadow v20261001120000     # start 7-day shadow
uv run python scripts/promote.py evaluate                   # live head-to-head
uv run python scripts/promote.py promote v20261001120000    # guarded by `evaluate`
uv run python scripts/promote.py promote v20261001120000 --force   # first deploy
uv run python scripts/promote.py rollback
"""

import argparse
import json

from preflight_core.db.engine import get_engine
from preflight_core.db.schema import model_runs
from preflight_core.models.promotion import compare_live
from preflight_core.models.registry import get_registry


def set_status(version: str, status: str) -> None:
    with get_engine().begin() as c:
        c.execute(model_runs.update().where(model_runs.c.version == version).values(status=status))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "shadow", "evaluate", "promote", "rollback"])
    ap.add_argument("version", nargs="?")
    ap.add_argument("--force", action="store_true", help="skip the live comparison")
    args = ap.parse_args()
    reg = get_registry()
    ptr = reg.read()

    if args.action == "status":
        print(json.dumps(ptr, indent=2))
    elif args.action == "shadow":
        reg.set_shadow(args.version)
        set_status(args.version, "shadow")
        print(f"{args.version} is now shadow (scored, hidden)")
    elif args.action == "evaluate":
        if not (ptr["champion"] and ptr["shadow"]):
            raise SystemExit("need both a champion and a shadow")
        print(
            json.dumps(
                compare_live(get_engine(), ptr["champion"], ptr["shadow"]), indent=2, default=str
            )
        )
    elif args.action == "promote":
        if ptr["champion"] and not args.force:
            verdict = compare_live(get_engine(), ptr["champion"], args.version)
            if not verdict["promote"]:
                raise SystemExit(f"not promoting: {verdict['reason']}")
        old = ptr["champion"]
        reg.promote(args.version)
        set_status(args.version, "champion")
        if old and old != args.version:
            set_status(old, "retired")
        print(f"{args.version} is champion (previous: {old}); scorers pick it up within 30 s")
    elif args.action == "rollback":
        bad = ptr["champion"]
        restored = reg.rollback()
        set_status(restored, "champion")
        if bad:
            set_status(bad, "rolled_back")
        print(f"rolled back {bad} -> {restored}")


if __name__ == "__main__":
    main()
