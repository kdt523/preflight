"""Counterfactual what-ifs (SPEC feature 3): edit actionable features, re-score, show the delta.

Edits change every feature the action would really change, including size: adding tests also
adds lines. A scenario is only suggested if the calibrated risk actually drops.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

Edit = Callable[[dict[str, float], dict[str, Any]], dict[str, float] | None]
MIN_DELTA = 0.01


def _add_tests(f, d):
    untested = d.get("untested_files") or []
    if f["touches_tests"] and not untested:
        return None
    code_added = f["lines_added"] * (1 - f["frac_test_files"])
    new_tests = max(1, len(untested))
    added_lines = max(10.0, 0.5 * code_added)
    n_files = f["n_files"] + new_tests
    g = dict(f)
    g.update(
        touches_tests=1.0,
        frac_src_with_tests=1.0,
        test_code_ratio=max(f["test_code_ratio"], added_lines / max(1.0, code_added)),
        n_files=n_files,
        lines_added=f["lines_added"] + added_lines,
        log_churn=math.log1p(f["lines_added"] + added_lines + f["lines_deleted"]),
        frac_test_files=(f["frac_test_files"] * f["n_files"] + new_tests) / n_files,
    )
    return g


def _rebase(f, d):
    if not (f["pushes_since_base"] == f["pushes_since_base"] and f["pushes_since_base"] > 0):
        return None
    return dict(f, pushes_since_base=0.0, concurrent_same_files=f["concurrent_same_files"])


def _split(f, d):
    if f["n_files"] < 6:
        return None
    half = lambda x: math.ceil(x / 2)  # noqa: E731
    added, deleted = f["lines_added"] / 2, f["lines_deleted"] / 2
    return dict(
        f,
        n_files=half(f["n_files"]),
        n_dirs=half(f["n_dirs"]),
        lines_added=added,
        lines_deleted=deleted,
        log_churn=math.log1p(added + deleted),
    )


def _include_partners(f, d):
    if not f["missing_partners"]:
        return None
    k = f["missing_partners"]
    return dict(f, missing_partners=0.0, max_missing_confidence=0.0, n_files=f["n_files"] + k)


SCENARIOS: dict[str, tuple[str, Edit]] = {
    "add_tests": ("Add or update tests for the changed code", _add_tests),
    "include_partners": ("Update the files that usually change with these", _include_partners),
    "rebase": ("Rebase onto the latest default branch", _rebase),
    "split": ("Split this into two smaller PRs", _split),
}


@dataclass(frozen=True)
class WhatIf:
    scenario: str
    text: str
    p_before: float
    p_after: float

    @property
    def delta(self) -> float:
        return self.p_after - self.p_before


def what_ifs(
    predict: Callable[[pd.DataFrame], Any],
    features: dict[str, float],
    details: dict[str, Any],
    columns: list[str],
) -> list[WhatIf]:
    base = pd.DataFrame([features])[columns]
    edited: list[tuple[str, str, dict[str, float]]] = []
    for key, (text, edit) in SCENARIOS.items():
        g = edit(features, details)
        if g is not None:
            edited.append((key, text, g))
    if not edited:
        return []
    probs = predict(
        pd.concat([base, pd.DataFrame([g for _, _, g in edited])[columns]], ignore_index=True)
    )
    p0 = float(probs[0])
    out = [WhatIf(k, t, p0, float(p)) for (k, t, _), p in zip(edited, probs[1:], strict=True)]
    return sorted((w for w in out if w.delta <= -MIN_DELTA), key=lambda w: w.delta)
