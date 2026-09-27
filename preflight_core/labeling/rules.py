"""Labeler v0: failure taxonomy from rerun history, job logs and failed-step names (SPEC §4).

Pure functions only. Every label carries its source and the rule that fired, so label
quality can be measured per source against hand labels (week 2).

Precedence for a failed build:
  1. a later run of the same sha/workflow/event passed           -> flaky   (source: rerun)
  2. GitHub conclusion timed_out / startup_failure               -> infra / config
  3. per failed job, from its log if retained, else its step name:
       strong infra > config > code evidence > weak infra > unknown
  4. build = code_defect if any job shows code evidence, else config, else infra,
     else code_defect with source 'default' (unexplained failure; can be filtered later)
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

PASS, CODE, FLAKY, INFRA, CONFIG, CANCELLED = (
    "pass",
    "code_defect",
    "flaky",
    "infra",
    "config",
    "cancelled",
)
CLASSES_FOR_LOGS = (CODE, FLAKY, INFRA, CONFIG)  # hand-label / log-classifier classes
DROPPED_CONCLUSIONS = {"cancelled", "skipped", "neutral", "stale", "action_required"}


_NET_EXC = (
    r"(?:Connection|Timeout|ReadTimeout|ConnectTimeout|SSL|ProxyError|MaxRetry|RemoteDisconnected)"
)


def _rules(pairs: list[tuple[str, str]]) -> list[tuple[str, re.Pattern[str]]]:
    return [(name, re.compile(rx, re.I)) for name, rx in pairs]


STRONG_INFRA = _rules(
    [
        ("runner_shutdown", r"runner has received a shutdown signal"),
        (
            "runner_lost",
            r"lost communication with the server|runner .* did not connect|"
            r"the hosted runner .* lost communication",
        ),
        (
            "oom",
            r"out of memory|OOMKilled|cannot allocate memory|exit code 137\b|"
            r"heap out of memory|\bKilled\s*$",
        ),
        ("disk_full", r"no space left on device"),
        ("max_time", r"exceeded the maximum execution time|has exceeded the maximum"),
        ("runner_image", r"unable to (?:pull|find) image|failed to initialize container"),
    ]
)

CONFIG = _rules(
    [
        (
            "invalid_workflow",
            r"invalid workflow file|unable to resolve action|"
            r"can't find 'action\.ya?ml'",
        ),
        (
            "pip_resolution",
            r"ResolutionImpossible|could not find a version that satisfies|"
            r"no matching distribution found",
        ),
        (
            "resolver",
            r"version solving failed|npm ERR! (?:code )?ERESOLVE|"
            r"unable to resolve dependency tree|conflicting dependencies",
        ),
        (
            "lockfile",
            r"lock file .* (?:is )?(?:not up to date|out of date)|"
            r"pyproject\.toml changed significantly since poetry\.lock",
        ),
    ]
)

CODE_EVIDENCE = _rules(
    [
        ("pytest_failed", r"=+ .*\b\d+ failed\b|^FAILED \S+|^ERROR \S+::"),
        ("assertion", r"\bAssertionError\b|^E\s{2,}assert\b"),
        (
            "python_error",
            r"^(?:\w+\.)*(?!" + _NET_EXC + r")\w+(?:Error|Exception): |SyntaxError"
            r"|ImportError|ModuleNotFoundError",
        ),
        ("type_check", r"Found \d+ errors? in \d+ files?|error: .*\[[a-z-]+\]$"),
        ("lint", r"Found \d+ errors?\.|would reformat|^\S+:\d+:\d+: [EFW]\d{3} "),
        ("compile", r"compilation (?:failed|terminated)|error\[E\d+\]|: error: "),
        ("test_summary", r"\bTests? (?:run|failed):|\d+ tests? failed|FAILURES?\b"),
    ]
)

WEAK_INFRA = _rules(
    [
        (
            "network",
            r"could not resolve host|temporary failure in name resolution|"
            r"connection reset by peer|ECONNRESET|ETIMEDOUT|TLS handshake timeout|"
            r"failed to connect to|ReadTimeoutError|\b50[234] (?:bad gateway|service unavailable|"
            r"gateway time-?out)",
        ),
        ("network_exception", r"^(?:\w+\.)*" + _NET_EXC + r"\w*: "),
        ("rate_limit", r"rate limit exceeded|toomanyrequests|429 too many requests"),
        ("cancelled_op", r"the operation was canceled"),
    ]
)

STEP_INFRA = re.compile(
    r"^(set up job|initialize containers|complete job|set up runner)$|checkout|cache", re.I
)
STEP_CONFIG = re.compile(
    r"install|dependenc|setup-python|set up python|setup-node|set up node|uv sync|poetry", re.I
)
STEP_CODE = re.compile(
    r"test|pytest|tox|nox|lint|ruff|flake8|mypy|pyright|type|build|compile|check|format", re.I
)


@dataclass(frozen=True)
class Label:
    failure_type: str
    source: str
    reason: str


@dataclass(frozen=True)
class JobEvidence:
    conclusion: str | None
    failed_step: str | None
    log_lines: list[str] | None  # redacted tail; None if not retained / not fetched


def _first_match(rules, lines: list[str]) -> str | None:
    for name, rx in rules:
        if any(rx.search(ln) for ln in lines):
            return name
    return None


LogModel = Callable[[list[str]], "tuple[str, float] | None"]


def classify_job(
    job: JobEvidence, log_model: LogModel | None = None
) -> tuple[str | None, str, str]:
    """-> (category or None if unknown, source, reason).

    `log_model` (the TF-IDF classifier) is consulted only when rules find nothing in a log."""
    if job.conclusion == "timed_out":
        return INFRA, "conclusion", "job_timed_out"
    if job.log_lines:
        lines = job.log_lines
        for cat, rules in (
            (INFRA, STRONG_INFRA),
            (CONFIG, CONFIG),
            (CODE, CODE_EVIDENCE),
            (INFRA, WEAK_INFRA),
        ):
            if hit := _first_match(rules, lines):
                return cat, "log", hit
        if log_model and (pred := log_model(lines)):
            return pred[0], "classifier", f"clf:{pred[1]:.2f}"
    if step := job.failed_step:
        if STEP_INFRA.search(step):
            return INFRA, "step", f"step:{step}"
        if STEP_CONFIG.search(step):
            return CONFIG, "step", f"step:{step}"
        if STEP_CODE.search(step):
            return CODE, "step", f"step:{step}"
    return None, "none", "unexplained"


def classify_build(
    conclusion: str | None,
    later_success: bool,
    jobs: list[JobEvidence],
    log_model: LogModel | None = None,
) -> Label:
    if conclusion is None or conclusion in DROPPED_CONCLUSIONS:
        return Label(CANCELLED, "conclusion", conclusion or "no_conclusion")
    if conclusion == "success":
        return Label(PASS, "conclusion", "success")
    if later_success:
        return Label(FLAKY, "rerun", "same sha/workflow passed later")
    if conclusion == "timed_out":
        return Label(INFRA, "conclusion", "run_timed_out")
    if conclusion == "startup_failure":
        return Label(CONFIG, "conclusion", "startup_failure")

    failed = [j for j in jobs if j.conclusion in ("failure", "timed_out")]
    results = [classify_job(j, log_model) for j in failed]
    for cat in (CODE, CONFIG, INFRA, FLAKY):
        hits = [r for r in results if r[0] == cat]
        if hits:
            return Label(cat, hits[0][1], hits[0][2])
    return Label(CODE, "default", "unexplained failure")
