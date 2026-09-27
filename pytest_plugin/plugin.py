"""pytest plugin: run the tests most likely to fail first (SPEC feature 4).

    pytest --preflight                      # uses PREFLIGHT_URL / git remote for the repo
    pytest --preflight --preflight-url http://scorer:8001 --preflight-base origin/main

Fail-open: any error or a response slower than --preflight-timeout keeps pytest's own order.
Only file paths and line counts leave the machine, never source code.
"""

import os

import httpx
import pytest

from preflight_core.localgit import local_diff, repo_slug


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("preflight", "Preflight fail-fast test ordering")
    g.addoption("--preflight", action="store_true", help="reorder tests by predicted failure")
    g.addoption("--preflight-url", default=os.environ.get("PREFLIGHT_URL", "http://localhost:8001"))
    g.addoption("--preflight-repo", default=os.environ.get("PREFLIGHT_REPO"))
    g.addoption("--preflight-base", default=os.environ.get("PREFLIGHT_BASE", "origin/main"))
    g.addoption("--preflight-timeout", type=float, default=2.0)


def reorder(items: list, order: list[str]) -> list:
    """Items in `order`; anything the server didn't return keeps its place at the end."""
    rank = {nid: i for i, nid in enumerate(order)}
    pos = {id(it): i for i, it in enumerate(items)}
    return sorted(items, key=lambda it: (rank.get(it.nodeid, len(rank)), pos[id(it)]))


def pytest_collection_modifyitems(session, config: pytest.Config, items: list) -> None:
    if not config.getoption("preflight") or len(items) < 2:
        return
    tr = config.pluginmanager.get_plugin("terminalreporter")

    def say(msg: str) -> None:
        if tr:
            tr.write_line(f"preflight: {msg}")

    try:
        repo = config.getoption("preflight_repo") or repo_slug(config.rootpath)
        if not repo:
            say("ordering skipped (repo unknown; pass --preflight-repo)")
            return
        changed = [
            f["path"] for f in local_diff(config.getoption("preflight_base"), config.rootpath)
        ]
        r = httpx.post(
            f"{config.getoption('preflight_url').rstrip('/')}/v1/tests/order",
            json={"repo": repo, "changed_files": changed, "tests": [i.nodeid for i in items]},
            timeout=config.getoption("preflight_timeout"),
            headers={"Authorization": f"Bearer {tok}"}
            if (tok := os.environ.get("PREFLIGHT_TOKEN"))
            else {},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # fail-open: never break or slow a test run
        say(f"ordering skipped ({type(e).__name__}); running in default order")
        return
    if data.get("reordered"):
        items[:] = reorder(items, data["order"])
        say(f"reordered {len(items)} tests; first: {items[0].nodeid}")
    else:
        say("no ranking model for this repo yet; default order")
