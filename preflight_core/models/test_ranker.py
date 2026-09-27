"""Per-test failure model for fail-fast ordering (SPEC feature 4). Metric: APFD.

P(test fails | this change) from point-in-time history:
  - the test's own smoothed failure rate and its last-5 outcomes
  - co-failure: how often it failed when each changed file changed
  - path similarity between the test and the changed files
Ranking = P(fail) descending, cheaper tests first on ties.
"""

import math
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from preflight_core.features.state import LeakageError, smoothed
from preflight_core.models.metrics import apfd

TEST_FEATURES = [
    "t_fail_rate",
    "t_recent_fails",
    "t_log_runs",
    "t_is_new",
    "cofail_max",
    "cofail_files",
    "path_sim",
    "name_match",
]
TEST_MONOTONE = [1, 1, 0, 0, 1, 1, 0, 0]
FAILED = {"failed", "error"}
TEST_PRIOR_STRENGTH = 5.0
COFAIL_PRIOR_STRENGTH = 5.0
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"test", "tests", "py", "src", "lib", "unit", "integration", "the", "init"}


def canonical_test_id(test_id: str) -> str:
    """JUnit 'tests.test_api.TestX::test_y' and pytest 'tests/test_api.py::TestX::test_y'
    both become 'tests/test_api.py::TestX::test_y'."""
    if ".py::" in test_id or "::" not in test_id:
        return test_id
    cls, name = test_id.split("::", 1)
    parts = cls.split(".")
    classes = []
    while parts and parts[-1][:1].isupper():
        classes.insert(0, parts.pop())
    path = "/".join(parts) + ".py" if parts else cls
    return "::".join([path, *classes, name])


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN.findall(s.lower()) if t not in _STOP and len(t) > 2}


@dataclass
class TestRun:
    key: str
    as_of: float
    completed_at: float
    changed_files: list[str]
    results: dict[str, tuple[str, int | None]]  # test_id -> (status, duration_ms)


@dataclass
class TestHistory:
    watermark: float = float("-inf")
    runs: int = 0
    fails: int = 0
    stats: dict[str, list[int]] = field(default_factory=dict)  # test -> [runs, fails]
    recent: dict[str, deque] = field(default_factory=dict)
    duration: dict[str, float] = field(default_factory=dict)
    file_runs: Counter = field(default_factory=Counter)
    cofail: dict[str, Counter] = field(default_factory=dict)

    def prior(self) -> float:
        return smoothed(self.fails, self.runs, 0.01, 100.0)

    def features(self, test_id: str, changed: list[str], as_of: float) -> dict[str, float]:
        if self.watermark >= as_of:
            raise LeakageError("test history read at/after as_of")
        prior = self.prior()
        s = self.stats.get(test_id)
        cof = [
            smoothed(
                self.cofail.get(f, {}).get(test_id, 0),
                self.file_runs[f],
                prior,
                COFAIL_PRIOR_STRENGTH,
            )
            for f in changed
            if self.file_runs[f]
        ]
        ttok = _tokens(test_id)
        ctok = set().union(*(_tokens(f) for f in changed)) if changed else set()
        stems = {f.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() for f in changed}
        return {
            "t_fail_rate": smoothed(s[1], s[0], prior, TEST_PRIOR_STRENGTH) if s else prior,
            "t_recent_fails": float(sum(self.recent.get(test_id, ()))),
            "t_log_runs": math.log1p(s[0]) if s else 0.0,
            "t_is_new": float(s is None),
            "cofail_max": max(cof, default=prior),
            "cofail_files": float(sum(c > 2 * prior for c in cof)),
            "path_sim": len(ttok & ctok) / len(ttok | ctok) if ttok | ctok else 0.0,
            "name_match": float(any(st and st in test_id.lower() for st in stems)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "watermark": self.watermark,
            "runs": self.runs,
            "fails": self.fails,
            "stats": self.stats,
            "recent": {k: list(v) for k, v in self.recent.items()},
            "duration": self.duration,
            "file_runs": dict(self.file_runs),
            "cofail": {k: dict(v) for k, v in self.cofail.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestHistory":
        return cls(
            watermark=d["watermark"],
            runs=d["runs"],
            fails=d["fails"],
            stats={k: list(v) for k, v in d["stats"].items()},
            recent={k: deque(v, maxlen=5) for k, v in d["recent"].items()},
            duration=dict(d["duration"]),
            file_runs=Counter(d["file_runs"]),
            cofail={k: Counter(v) for k, v in d["cofail"].items()},
        )

    def apply(self, run: TestRun) -> None:
        self.watermark = max(self.watermark, run.completed_at)
        failed_here = {t for t, (st, _) in run.results.items() if st in FAILED}
        for f in run.changed_files:
            self.file_runs[f] += 1
            if failed_here:
                self.cofail.setdefault(f, Counter()).update(failed_here)
        for t, (st, dur) in run.results.items():
            if st == "skipped":
                continue
            fail = int(st in FAILED)
            s = self.stats.setdefault(t, [0, 0])
            s[0] += 1
            s[1] += fail
            self.runs += 1
            self.fails += fail
            self.recent.setdefault(t, deque(maxlen=5)).append(fail)
            if dur is not None:
                self.duration[t] = 0.7 * self.duration.get(t, dur) + 0.3 * dur


def replay_tests(runs: list[TestRun], neg_per_run: int | None = None, seed: int = 0):
    """-> (rows DataFrame, per-run eval records). Outcomes apply at completion time."""
    rng = random.Random(seed)
    events = sorted(
        [(r.as_of, 0, i) for i, r in enumerate(runs)]
        + [(r.completed_at, 1, i) for i, r in enumerate(runs)]
    )
    hist = TestHistory()
    rows, evals = [], []
    for _, kind, i in events:
        r = runs[i]
        if kind == 1:
            hist.apply(r)
            continue
        tests = [t for t, (st, _) in r.results.items() if st != "skipped"]
        feats = {t: hist.features(t, r.changed_files, r.as_of) for t in tests}
        failing = {t for t in tests if r.results[t][0] in FAILED}
        evals.append(
            {
                "key": r.key,
                "tests": tests,
                "failing": failing,
                "features": feats,
                "durations": {t: r.results[t][1] or 0 for t in tests},
            }
        )
        keep = tests
        if neg_per_run is not None:
            neg = [t for t in tests if t not in failing]
            keep = list(failing) + rng.sample(neg, min(neg_per_run, len(neg)))
        for t in keep:
            rows.append({"run": r.key, "test_id": t, "y": int(t in failing), **feats[t]})
    return pd.DataFrame(rows, columns=["run", "test_id", "y", *TEST_FEATURES]), evals


class TestRanker:
    def fit(self, rows: pd.DataFrame) -> "TestRanker":
        y = rows["y"].to_numpy()
        pos = max(1, y.sum())
        self.clf_ = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            monotone_constraints=TEST_MONOTONE,
            scale_pos_weight=(len(y) - pos) / pos,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
            random_state=0,
        ).fit(rows[TEST_FEATURES], y)
        return self

    def scores(self, feats: dict[str, dict[str, float]]) -> dict[str, float]:
        if not feats:
            return {}
        tests = list(feats)
        X = pd.DataFrame([feats[t] for t in tests])[TEST_FEATURES]
        return dict(zip(tests, self.clf_.predict_proba(X)[:, 1], strict=True))

    def rank(
        self, feats: dict[str, dict[str, float]], durations: dict[str, float] | None = None
    ) -> list[str]:
        s = self.scores(feats)
        durations = durations or {}
        return sorted(s, key=lambda t: (-round(s[t], 6), durations.get(t, 0), t))


def time_to_first_failure(
    order: list[str], failing: set[str], durations: dict[str, float]
) -> float:
    total = 0.0
    for t in order:
        total += durations.get(t, 0)
        if t in failing:
            return total
    return total


def evaluate_ordering(ranker: TestRanker, evals: list[dict[str, Any]], seed: int = 0) -> dict:
    """APFD and time-to-first-failure: model vs default (alphabetical) vs random order."""
    rng = random.Random(seed)
    res = {"model": [], "default": [], "random": []}
    ttff = {"model": [], "default": []}
    for e in evals:
        if not e["failing"] or len(e["tests"]) < 2:
            continue
        orders = {
            "model": ranker.rank(e["features"], e["durations"]),
            "default": sorted(e["tests"]),
            "random": rng.sample(e["tests"], len(e["tests"])),
        }
        for k, o in orders.items():
            res[k].append(apfd(o, e["failing"]))
        for k in ttff:
            ttff[k].append(time_to_first_failure(orders[k], e["failing"], e["durations"]))
    if not res["model"]:
        return {"runs": 0}
    med_m, med_d = median(ttff["model"]), median(ttff["default"])
    return {
        "runs": len(res["model"]),
        "apfd_model": float(np.mean(res["model"])),
        "apfd_default": float(np.mean(res["default"])),
        "apfd_random": float(np.mean(res["random"])),
        "median_ttff_model_ms": med_m,
        "median_ttff_default_ms": med_d,
        "ttff_reduction": (1 - med_m / med_d) if med_d else float("nan"),
    }


def load_test_runs(engine, repo_id: int | None = None) -> list[TestRun]:
    from sqlalchemy import text

    where = "WHERE b.repo_id = :repo" if repo_id is not None else ""
    q = text(f"""
        SELECT b.id, b.repo_id, b.sha, b.started_at, b.completed_at, t.test_id, t.status,
               t.duration_ms
        FROM test_results t JOIN builds b ON b.id = t.build_id {where}
        ORDER BY b.id
    """)
    files_q = text("SELECT repo_id, sha, path FROM file_changes")
    with engine.connect() as c:
        rows = c.execute(q, {"repo": repo_id}).all()
        files: dict[tuple[int, str], list[str]] = {}
        for r in c.execute(files_q):
            files.setdefault((r.repo_id, r.sha), []).append(r.path)
    runs: dict[int, TestRun] = {}
    for r in rows:
        run = runs.get(r.id)
        if run is None:
            run = runs[r.id] = TestRun(
                key=str(r.id),
                as_of=r.started_at.timestamp(),
                completed_at=(r.completed_at or r.started_at).timestamp(),
                changed_files=files.get((r.repo_id, r.sha), []),
                results={},
            )
        run.results[canonical_test_id(r.test_id)] = (r.status, r.duration_ms)
    return list(runs.values())
