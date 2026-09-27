"""Pick backfill repos by measured CI volume and failure rate, not by gut feel."""

import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from preflight_core.github.client import GitHubClient, GitHubError

JUNIT_HINT = re.compile(r"junit|test[-_]?results?|pytest|test[-_]?reports?", re.I)


@dataclass
class RepoStats:
    full_name: str
    default_branch: str
    archived: bool
    runs: int
    failures: int
    junit_artifacts: bool

    @property
    def failure_rate(self) -> float:
        return self.failures / self.runs if self.runs else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "failure_rate": round(self.failure_rate, 4)}


def measure(gh: GitHubClient, full_name: str, window_days: int = 90) -> RepoStats:
    since = (datetime.now(UTC) - timedelta(days=window_days)).date().isoformat()
    repo = gh.get_json(f"/repos/{full_name}")
    runs_path = f"/repos/{full_name}/actions/runs"
    runs = gh.get_json(runs_path, {"created": f">={since}", "per_page": 1})["total_count"]
    fails = gh.get_json(runs_path, {"created": f">={since}", "status": "failure", "per_page": 1})[
        "total_count"
    ]
    artifacts = gh.get_json(f"/repos/{full_name}/actions/artifacts", {"per_page": 100})
    junit = any(JUNIT_HINT.search(a["name"]) for a in artifacts.get("artifacts", []))
    return RepoStats(
        full_name=repo["full_name"],
        default_branch=repo["default_branch"],
        archived=repo["archived"],
        runs=runs,
        failures=fails,
        junit_artifacts=junit,
    )


def eligible(s: RepoStats, min_runs: int = 500, rate_range=(0.02, 0.5)) -> bool:
    lo, hi = rate_range
    return not s.archived and s.runs >= min_runs and lo <= s.failure_rate <= hi


def score(s: RepoStats) -> float:
    # Volume matters most (log-scaled so one giant repo can't dominate);
    # failures give us positives; JUnit unlocks the per-test model.
    return (
        math.log10(max(s.runs, 1))
        + math.log10(max(s.failures, 1))
        + (1.0 if s.junit_artifacts else 0.0)
    )


def select(stats: list[RepoStats], n: int = 20) -> list[RepoStats]:
    return sorted((s for s in stats if eligible(s)), key=score, reverse=True)[:n]


def measure_all(gh: GitHubClient, names: list[str]) -> tuple[list[RepoStats], dict[str, str]]:
    stats, errors = [], {}
    for name in names:
        try:
            stats.append(measure(gh, name))
        except GitHubError as e:
            errors[name] = str(e)
    return stats, errors
