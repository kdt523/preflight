"""SHAP contributions -> plain-language reasons (SPEC feature 2).

Author-family features are removed before anything is shown: they can move the score but are
never displayed or returned by the API. Related features (e.g. the size family) collapse into
one sentence so the top reasons are distinct.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from preflight_core.features.definitions import HIDDEN_FAMILIES, family_of

Template = Callable[[dict[str, float], dict[str, Any]], str | None]


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _riskiest(f, d):
    r = d.get("riskiest_file")
    if not r or not r.get("builds"):
        return None
    return (
        f"`{r['path']}` failed {r['fails']} of the {r['builds']} builds that touched it "
        f"({_pct(r['rate'])} after smoothing; repo average {_pct(d.get('base_rate', 0))})."
    )


def _partners(f, d):
    mp = d.get("missing_partners")
    if not mp:
        return None
    partner, conf = mp[0]
    more = f" (+{len(mp) - 1} more)" if len(mp) > 1 else ""
    return (
        f"Files here usually change with `{partner}` ({_pct(conf)} of the time), "
        f"but it isn't in this change{more}."
    )


def _tests(f, d):
    untested = d.get("untested_files")
    if f.get("touches_tests", 0) == 0 and untested:
        return f"No tests changed, and `{untested[0]}` has no matching test file."
    if f.get("touches_tests", 0) == 0:
        return "No test files changed alongside the code."
    if untested:
        return f"`{untested[0]}` changed without a matching test file."
    return "Tests were updated alongside the code."


def _size(f, d):
    return (
        f"Large change: {int(f['n_files'])} files, +{int(f['lines_added'])}/"
        f"-{int(f['lines_deleted'])} lines across {int(f['n_dirs'])} directories."
    )


def _size_small(f, d):
    lines = int(f["lines_added"] + f["lines_deleted"])
    return f"Small, focused change ({int(f['n_files'])} files, {lines} lines)."


TEMPLATES: dict[str, tuple[str, Template, Template | None]] = {
    # feature -> (group, risk-raising sentence, risk-lowering sentence)
    "max_file_fail_rate": ("file", _riskiest, None),
    "mean_file_fail_rate": ("file", _riskiest, None),
    "file_rate_lift": ("file", _riskiest, None),
    "hotspot_score": ("file", _riskiest, None),
    "missing_partners": ("cochange", _partners, None),
    "max_missing_confidence": ("cochange", _partners, None),
    "touches_tests": ("tests", _tests, _tests),
    "test_code_ratio": ("tests", _tests, _tests),
    "frac_src_with_tests": ("tests", _tests, _tests),
    "n_files": ("size", _size, _size_small),
    "lines_added": ("size", _size, _size_small),
    "lines_deleted": ("size", _size, _size_small),
    "log_churn": ("size", _size, _size_small),
    "n_dirs": ("size", _size, _size_small),
    "change_entropy": (
        "size",
        lambda f, d: (
            f"Changes are spread thinly across {int(f['n_files'])} files (entropy "
            f"{f['change_entropy']:.2f})."
        ),
        None,
    ),
    "touches_lockfile": (
        "deps",
        lambda f, d: "A dependency lockfile changed." if f["touches_lockfile"] else None,
        None,
    ),
    "touches_dep_manifest": (
        "deps",
        lambda f, d: "Dependency declarations changed." if f["touches_dep_manifest"] else None,
        None,
    ),
    "touches_ci_config": (
        "ci",
        lambda f, d: "CI configuration changed." if f["touches_ci_config"] else None,
        None,
    ),
    "touches_migrations": (
        "migr",
        lambda f, d: "A database migration changed." if f["touches_migrations"] else None,
        None,
    ),
    "main_red": (
        "main",
        lambda f, d: "The default branch is currently failing." if f["main_red"] else None,
        lambda f, d: "The default branch is green.",
    ),
    "hours_since_last_green": (
        "main",
        lambda f, d: (
            (f"The default branch last passed {f['hours_since_last_green']:.0f} h ago.")
            if f["hours_since_last_green"] == f["hours_since_last_green"]
            else None
        ),
        None,
    ),
    "pushes_since_base": (
        "rebase",
        lambda f, d: (
            (
                f"{int(f['pushes_since_base'])} pushes landed on the default branch "
                "since this PR's base."
            )
            if f["pushes_since_base"] == f["pushes_since_base"] and f["pushes_since_base"] > 0
            else None
        ),
        None,
    ),
    "concurrent_same_files": (
        "concurrent",
        lambda f, d: (
            (f"{int(f['concurrent_same_files'])} other in-flight builds touch the same files.")
            if f["concurrent_same_files"]
            else None
        ),
        None,
    ),
    "repo_recent_fail_rate": (
        "repo",
        lambda f, d: (
            f"Recent builds in this repo fail {_pct(f['repo_recent_fail_rate'])} of the time."
        ),
        None,
    ),
    "docs_only": (
        "docs",
        None,
        lambda f, d: "Only documentation changed." if f["docs_only"] else None,
    ),
}


@dataclass(frozen=True)
class Reason:
    feature: str
    contribution: float  # log-odds
    direction: str  # "raises" | "lowers"
    text: str


def reasons_for_row(
    features: dict[str, float],
    details: dict[str, Any],
    contributions: dict[str, float],
    top_k: int = 3,
) -> list[Reason]:
    ranked = sorted(
        ((n, c) for n, c in contributions.items() if family_of(n) not in HIDDEN_FAMILIES),
        key=lambda nc: -abs(nc[1]),
    )
    out: list[Reason] = []
    used_groups: set[str] = set()
    for name, c in ranked:
        if abs(c) < 1e-3 or name not in TEMPLATES:
            continue
        group, up, down = TEMPLATES[name]
        tmpl = up if c > 0 else down
        if tmpl is None or group in used_groups:
            continue
        text = tmpl(features, details)
        if not text:
            continue
        used_groups.add(group)
        out.append(Reason(name, float(c), "raises" if c > 0 else "lowers", text))
        if len(out) == top_k:
            break
    return out


def explain(model, X: pd.DataFrame, details: list[dict[str, Any]], top_k: int = 3):
    """Reasons for each row of X (model = CalibratedLGBM)."""
    contrib = model.contributions(X)[:, :-1]
    names = model.features
    out = []
    for i in range(len(X)):
        feats = {n: float(X.iloc[i][n]) for n in names}
        c = {n: float(v) for n, v in zip(names, contrib[i], strict=True)}
        out.append(reasons_for_row(feats, details[i], c, top_k))
    return out


def visible_contributions(names: list[str], contrib: np.ndarray) -> dict[str, float]:
    """Contributions safe to return over the API (author family removed)."""
    return {
        n: float(v)
        for n, v in zip(names, contrib, strict=False)
        if family_of(n) not in HIDDEN_FAMILIES
    }
