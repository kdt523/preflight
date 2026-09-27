"""Point-in-time history state for one repo.

Replays two kinds of events per build unit:
  start      at as_of        -> the change is known (churn, co-change, author touches, in-flight)
  complete   at completed_at -> the outcome is known (file/author/repo fail rates, main red/green)

`features(unit)` may only read state built from events strictly before unit.as_of. The
state records a watermark of the latest event applied and refuses to compute features if
the watermark is at/after as_of: the leakage guard is enforced in code, not by convention.

The same class backs offline dataset building and the online store (serialised to Redis),
so training and serving compute features with identical code.
"""

import bisect
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from preflight_core.features.diff import FileChange, diff_features, test_basenames_for

TAU_CHURN_S = 30 * 86400  # decayed churn time constant
GLOBAL_PRIOR_RATE = 0.10  # prior for a repo's base rate before it has history
REPO_PRIOR_STRENGTH = 20.0
FILE_PRIOR_STRENGTH = 10.0  # m in (fails + m p) / (builds + m)
AUTHOR_PRIOR_STRENGTH = 10.0
COCHANGE_MIN_SUPPORT = 5
COCHANGE_MIN_CONFIDENCE = 0.6
COCHANGE_MAX_FILES = 50  # large refactors are noise for co-change mining
INFLIGHT_WINDOW_S = 24 * 3600
RECENT_N = 50


class LeakageError(AssertionError):
    pass


@dataclass
class Unit:
    """One prediction unit: a commit's first CI outcome in a repo."""

    key: str  # sha
    as_of: float  # epoch seconds: first CI start for this sha
    completed_at: float | None
    label: int | None  # 1 code_defect, 0 pass, None excluded (flaky/infra/config) or unknown
    files: list[FileChange]
    truncated: bool = False
    author: str | None = None
    is_pr: bool = False
    is_default_branch: bool = False
    base_sha: str | None = None


def smoothed(fails: float, builds: float, prior: float, strength: float) -> float:
    """Bayesian (beta-binomial) smoothing: (fails + a) / (builds + a + b), a + b = strength."""
    return (fails + strength * prior) / (builds + strength)


@dataclass
class HistoryState:
    watermark: float = float("-inf")
    repo_builds: int = 0
    repo_fails: int = 0
    # path -> [builds, fails, decayed_churn, churn_ts]
    files: dict[str, list[float]] = field(default_factory=dict)
    # co-change: path -> times changed (in eligible units), path -> Counter(partner)
    co_count: Counter = field(default_factory=Counter)
    co_pairs: dict[str, Counter] = field(default_factory=dict)
    test_basenames: set[str] = field(default_factory=set)
    # author_hash -> [builds_started, labelled_builds, fails]
    authors: dict[str, list[int]] = field(default_factory=dict)
    author_files: dict[str, set[str]] = field(default_factory=dict)
    # default branch
    default_push_times: list[float] = field(default_factory=list)  # sorted
    default_sha_time: dict[str, float] = field(default_factory=dict)
    last_green: float | None = None
    main_red: int = 0
    # in-flight: key -> (as_of, files)
    inflight: dict[str, tuple[float, frozenset[str]]] = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))

    # -- reading ---------------------------------------------------------------

    def base_rate(self) -> float:
        return smoothed(self.repo_fails, self.repo_builds, GLOBAL_PRIOR_RATE, REPO_PRIOR_STRENGTH)

    def file_rate(self, path: str, prior: float) -> float:
        s = self.files.get(path)
        return smoothed(s[1], s[0], prior, FILE_PRIOR_STRENGTH) if s else prior

    def decayed_churn(self, path: str, t: float) -> float:
        s = self.files.get(path)
        if not s:
            return 0.0
        return s[2] * math.exp(-(t - s[3]) / TAU_CHURN_S)

    def features(self, u: Unit) -> tuple[dict[str, float], dict[str, Any]]:
        if self.watermark >= u.as_of:
            raise LeakageError(f"state at {self.watermark} read for as_of {u.as_of} ({u.key})")
        t = u.as_of
        paths = [f.path for f in u.files]
        pset = set(paths)
        feats = diff_features(u.files, u.truncated)
        details: dict[str, Any] = {}

        # file history
        prior = self.base_rate()
        rates = {p: self.file_rate(p, prior) for p in paths}
        churn = {p: self.decayed_churn(p, t) for p in paths}
        hot = {p: churn[p] * rates[p] for p in paths}
        if paths:
            risky = max(paths, key=lambda p: rates[p])
            s = self.files.get(risky, [0, 0, 0, 0])
            details["riskiest_file"] = {
                "path": risky,
                "rate": rates[risky],
                "builds": int(s[0]),
                "fails": int(s[1]),
            }
        max_rate = max(rates.values(), default=prior)
        feats |= {
            "file_rate_lift": max_rate / prior,
            "max_file_fail_rate": max_rate,
            "mean_file_fail_rate": (sum(rates.values()) / len(rates)) if rates else prior,
            "sum_decayed_churn": sum(churn.values()),
            "hotspot_score": max(hot.values(), default=0.0),
            "frac_files_unseen": (sum(p not in self.files for p in paths) / len(paths))
            if paths
            else 0.0,
        }

        # co-change: rules f -> g with enough support and confidence, g absent from this change
        missing: dict[str, float] = {}
        for p in paths:
            n = self.co_count.get(p, 0)
            if n < COCHANGE_MIN_SUPPORT:
                continue
            for g, c in self.co_pairs.get(p, {}).items():
                conf = c / n
                if g not in pset and conf >= COCHANGE_MIN_CONFIDENCE:
                    missing[g] = max(missing.get(g, 0.0), conf)
        feats["missing_partners"] = len(missing)
        feats["max_missing_confidence"] = max(missing.values(), default=0.0)
        if missing:
            details["missing_partners"] = sorted(missing.items(), key=lambda kv: -kv[1])[:3]

        # test signal
        test_lines = sum(f.added for f in u.files if f.is_test)
        code_lines = sum(f.added for f in u.files if not f.is_test)
        src = [f.path for f in u.files if not f.is_test and _is_code(f.path)]
        known_tests = self.test_basenames | {_base(f.path) for f in u.files if f.is_test}
        untested = [p for p in src if not (test_basenames_for(p) & known_tests)]
        feats |= {
            "test_code_ratio": test_lines / max(1, code_lines),
            "touches_tests": float(any(f.is_test for f in u.files)),
            "frac_src_with_tests": (1 - len(untested) / len(src)) if src else 1.0,
        }
        if untested:
            details["untested_files"] = untested[:3]

        # context
        base_t = self.default_sha_time.get(u.base_sha) if u.base_sha else None
        feats |= {
            "is_pr": float(u.is_pr),
            "is_default_branch": float(u.is_default_branch),
            "pushes_since_base": float(
                len(self.default_push_times) - bisect.bisect_right(self.default_push_times, base_t)
            )
            if base_t is not None
            else math.nan,
            "main_red": float(self.main_red),
            "hours_since_last_green": (t - self.last_green) / 3600
            if self.last_green is not None
            else math.nan,
            "concurrent_same_files": float(
                sum(
                    1
                    for k, (ts, fs) in self.inflight.items()
                    if k != u.key and t - ts <= INFLIGHT_WINDOW_S and fs & pset
                )
            ),
            "repo_base_rate": prior,
            "repo_recent_fail_rate": smoothed(
                sum(self.recent), len(self.recent), prior, REPO_PRIOR_STRENGTH
            ),
        }

        # author (model only; never surfaced)
        a = self.authors.get(u.author or "", [0, 0, 0])
        touched = self.author_files.get(u.author or "", set())
        feats |= {
            "author_prior_builds": float(a[0]),
            "author_file_familiarity": (len(pset & touched) / len(pset)) if pset else 0.0,
            "author_fail_rate": smoothed(a[2], a[1], prior, AUTHOR_PRIOR_STRENGTH),
        }
        details["base_rate"] = prior
        return feats, details

    # -- writing ---------------------------------------------------------------

    def _advance(self, t: float) -> None:
        self.watermark = max(self.watermark, t)

    def apply_start(self, u: Unit) -> None:
        t = u.as_of
        self._advance(t)
        for f in u.files:
            s = self.files.setdefault(f.path, [0, 0, 0.0, t])
            s[2] = s[2] * math.exp(-(t - s[3]) / TAU_CHURN_S) + f.added + f.deleted
            s[3] = t
            if f.is_test:
                self.test_basenames.add(_base(f.path))
        paths = sorted({f.path for f in u.files})
        if 1 < len(paths) <= COCHANGE_MAX_FILES:
            for p in paths:
                self.co_count[p] += 1
                pairs = self.co_pairs.setdefault(p, Counter())
                for g in paths:
                    if g != p:
                        pairs[g] += 1
        elif len(paths) == 1:
            self.co_count[paths[0]] += 1
        if u.author:
            self.authors.setdefault(u.author, [0, 0, 0])[0] += 1
            self.author_files.setdefault(u.author, set()).update(paths)
        if u.is_default_branch:
            bisect.insort(self.default_push_times, t)
            self.default_sha_time.setdefault(u.key, t)
        self.inflight[u.key] = (t, frozenset(paths))
        self._prune_inflight(t)

    def apply_completion(self, u: Unit) -> None:
        assert u.completed_at is not None
        self._advance(u.completed_at)
        self.inflight.pop(u.key, None)
        if u.label is None:
            return
        self.repo_builds += 1
        self.repo_fails += u.label
        self.recent.append(u.label)
        for f in u.files:
            s = self.files.setdefault(f.path, [0, 0, 0.0, u.as_of])
            s[0] += 1
            s[1] += u.label
        if u.author:
            a = self.authors.setdefault(u.author, [0, 0, 0])
            a[1] += 1
            a[2] += u.label
        if u.is_default_branch:
            self.main_red = u.label
            if u.label == 0:
                self.last_green = u.completed_at

    def _prune_inflight(self, t: float) -> None:
        stale = [k for k, (ts, _) in self.inflight.items() if t - ts > INFLIGHT_WINDOW_S]
        for k in stale:
            del self.inflight[k]

    # -- serialisation (online store) -------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "watermark": self.watermark,
            "repo_builds": self.repo_builds,
            "repo_fails": self.repo_fails,
            "files": self.files,
            "co_count": dict(self.co_count),
            "co_pairs": {k: dict(v) for k, v in self.co_pairs.items()},
            "test_basenames": sorted(self.test_basenames),
            "authors": self.authors,
            "author_files": {k: sorted(v) for k, v in self.author_files.items()},
            "default_push_times": self.default_push_times,
            "default_sha_time": self.default_sha_time,
            "last_green": self.last_green,
            "main_red": self.main_red,
            "inflight": {k: [ts, sorted(fs)] for k, (ts, fs) in self.inflight.items()},
            "recent": list(self.recent),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HistoryState":
        return cls(
            watermark=d["watermark"],
            repo_builds=d["repo_builds"],
            repo_fails=d["repo_fails"],
            files={k: list(v) for k, v in d["files"].items()},
            co_count=Counter(d["co_count"]),
            co_pairs={k: Counter(v) for k, v in d["co_pairs"].items()},
            test_basenames=set(d["test_basenames"]),
            authors={k: list(v) for k, v in d["authors"].items()},
            author_files={k: set(v) for k, v in d["author_files"].items()},
            default_push_times=list(d["default_push_times"]),
            default_sha_time=dict(d["default_sha_time"]),
            last_green=d["last_green"],
            main_red=d["main_red"],
            inflight={k: (v[0], frozenset(v[1])) for k, v in d["inflight"].items()},
            recent=deque(d["recent"], maxlen=RECENT_N),
        )


def _event_groups(units: list[Unit]) -> list[tuple[float, list[Unit], list[Unit]]]:
    events: dict[float, tuple[list[Unit], list[Unit]]] = {}
    for u in units:
        events.setdefault(u.as_of, ([], []))[0].append(u)
        if u.completed_at is not None:
            if u.completed_at < u.as_of:
                raise ValueError(f"{u.key}: completed_at before as_of")
            events.setdefault(u.completed_at, ([], []))[1].append(u)
    return [(t, *events[t]) for t in sorted(events)]


def replay(units: list[Unit]) -> list[tuple[Unit, dict[str, float], dict[str, Any]]]:
    """Point-in-time features for every unit.

    Events are grouped by timestamp. Within a timestamp: compute features for every unit
    starting then, THEN apply those starts, THEN apply completions. So nothing at t is
    visible to a unit whose as_of is t.
    """
    state = HistoryState()
    out = []
    for _, starts, completions in _event_groups(units):
        for u in starts:
            feats, details = state.features(u)
            out.append((u, feats, details))
        for u in starts:
            state.apply_start(u)
        for u in completions:
            state.apply_completion(u)
    return out


def build_state(units: list[Unit], until: float) -> tuple[HistoryState, set[str], set[str]]:
    """State from every event strictly before `until` (the online store's snapshot).

    Also returns the unit keys whose start / completion were applied, so the live worker
    can apply later events incrementally without double-counting.
    """
    state = HistoryState()
    started: set[str] = set()
    completed: set[str] = set()
    for t, starts, completions in _event_groups(units):
        if t >= until:
            break
        for u in starts:
            state.apply_start(u)
            started.add(u.key)
        for u in completions:
            state.apply_completion(u)
            completed.add(u.key)
    return state, started, completed


def _base(path: str) -> str:
    return path.rsplit("/", 1)[-1]


_CODE_EXT = (
    ".py",
    ".pyx",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".scala",
    ".swift",
)


def _is_code(path: str) -> bool:
    return path.endswith(_CODE_EXT)
