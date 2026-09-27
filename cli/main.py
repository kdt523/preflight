"""preflight: score your local change before you push (SPEC feature 7).

    preflight check                 risk, reasons, what-ifs for HEAD + uncommitted edits
    preflight check --json          machine-readable
    preflight plan                  just the adaptive CI plan (for CI scripts)
    preflight install-hook          run `preflight check` on every git push

Sends only paths and line counts (never source) to PREFLIGHT_URL. Never blocks: exit code is
0 unless --strict is given and risk is high. Unreachable scorer -> a note, exit 0.
"""

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import httpx

from preflight_core.localgit import author_login, head_sha, local_diff, repo_slug

BAND_ICON = {"low": "🟢", "watch": "🟡", "high": "🔴"}
HOOK = """#!/bin/sh
# installed by `preflight install-hook`: informational, never blocks the push
preflight check --timeout 3 || true
"""


def _request(args: argparse.Namespace) -> dict:
    repo = args.repo or repo_slug()
    if not repo:
        raise SystemExit("preflight: can't tell the GitHub repo; pass --repo owner/name")
    files = local_diff(args.base)
    body = {
        "repo": repo,
        "sha": head_sha(),
        "files": files,
        "is_pr": True,
        "author_login": author_login(),  # hashed by the scorer, never stored raw
        "log_prediction": False,
    }
    headers = {"Authorization": f"Bearer {t}"} if (t := os.environ.get("PREFLIGHT_TOKEN")) else {}
    r = httpx.post(
        f"{args.url.rstrip('/')}/v1/score", json=body, headers=headers, timeout=args.timeout
    )
    r.raise_for_status()
    out = r.json()
    out["_files"] = len(files)
    return out


def render(s: dict) -> str:
    if s.get("abstain"):
        head = f"⚪ Not enough signal ({s['p_fail']:.0%} estimated risk)"
    else:
        head = f"{BAND_ICON[s['band']]} {s['band'].upper()} risk: {s['p_fail']:.0%} chance CI fails"
    lines = [head, f"   {s['_files']} files changed · CI plan: {s['ci_plan']}", ""]
    if s.get("reasons"):
        lines.append("Why")
        lines += [
            f"  {'↑' if r['direction'] == 'raises' else '↓'} {r['text']}" for r in s["reasons"]
        ]
        lines.append("")
    if s.get("what_ifs"):
        lines.append("What would lower it")
        lines += [
            f"  • {w['text']}: {w['p_before']:.0%} → {w['p_after']:.0%}" for w in s["what_ifs"]
        ]
        lines.append("")
    if s.get("test_order"):
        lines.append("Run these tests first")
        lines += [f"  {i}. {t}" for i, t in enumerate(s["test_order"][:5], 1)]
        lines.append("")
    lines.append(f"model {s['model_version']} · {s['latency_ms']:.0f} ms")
    return "\n".join(lines)


def cmd_check(args: argparse.Namespace) -> int:
    try:
        s = _request(args)
    except (httpx.HTTPError, OSError) as e:
        print(f"preflight: scorer unavailable ({type(e).__name__}); push proceeds as normal")
        return 0
    if args.json:
        print(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}, indent=2))
    else:
        print(render(s))
    return 1 if args.strict and s["band"] == "high" and not s.get("abstain") else 0


def cmd_plan(args: argparse.Namespace) -> int:
    try:
        print(_request(args)["ci_plan"])
    except (httpx.HTTPError, OSError):
        print("full")  # fail-open: unknown risk -> run everything
    return 0


def cmd_install_hook(args: argparse.Namespace) -> int:
    hooks = Path(".git/hooks")
    if not hooks.is_dir():
        raise SystemExit("preflight: run this from the root of a git repository")
    hook = hooks / "pre-push"
    if hook.exists() and "preflight" not in hook.read_text():
        raise SystemExit(f"preflight: {hook} exists; add `preflight check || true` to it yourself")
    hook.write_text(HOOK)
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {hook}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="preflight", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("plan", cmd_plan)):
        p = sub.add_parser(name)
        p.add_argument("--url", default=os.environ.get("PREFLIGHT_URL", "http://localhost:8001"))
        p.add_argument("--repo", default=os.environ.get("PREFLIGHT_REPO"))
        p.add_argument("--base", default=os.environ.get("PREFLIGHT_BASE", "origin/main"))
        p.add_argument("--timeout", type=float, default=5.0)
        if name == "check":
            p.add_argument("--json", action="store_true")
            p.add_argument(
                "--strict",
                action="store_true",
                help="exit 1 on high risk (opt-in; Preflight never blocks by default)",
            )
        p.set_defaults(fn=fn)
    sub.add_parser("install-hook").set_defaults(fn=cmd_install_hook)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
