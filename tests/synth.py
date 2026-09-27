"""Synthetic repo history FOR TESTS ONLY. Never used for any reported metric.

Planted signals the features/models should recover:
  - src/core/engine.py and src/core/parser.py fail far more often than other files
  - src/api/views.py and src/api/urls.py are almost always changed together
  - bigger, untested changes fail more
"""

import math
import random

from preflight_core.features.diff import FileChange
from preflight_core.features.state import Unit

RISKY = ["src/core/engine.py", "src/core/parser.py"]
PAIR = ["src/api/views.py", "src/api/urls.py"]
SAFE = [f"src/util/mod{i}.py" for i in range(20)] + ["README.md", "docs/index.rst"]
T0 = 1_760_000_000.0  # fixed epoch for reproducibility


def _sha(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(40))


def make_units(n: int = 1500, seed: int = 0, days: float = 180) -> list[Unit]:
    rng = random.Random(seed)
    units: list[Unit] = []
    default_shas: list[str] = []
    t = T0
    step = days * 86400 / n
    for _ in range(n):
        t += rng.expovariate(1 / step)
        files: list[FileChange] = []
        touched_risky = rng.random() < 0.2
        if touched_risky:
            files.append(FileChange(rng.choice(RISKY), rng.randint(5, 80), rng.randint(0, 40)))
        missing_partner = False
        if rng.random() < 0.25:
            files.append(FileChange(PAIR[0], rng.randint(1, 30), rng.randint(0, 10)))
            if rng.random() < 0.85:
                files.append(FileChange(PAIR[1], rng.randint(1, 5), rng.randint(0, 3)))
            else:
                missing_partner = True
        for p in rng.sample(SAFE, rng.randint(0 if files else 1, 4)):
            files.append(FileChange(p, rng.randint(1, 60), rng.randint(0, 30)))
        has_tests = rng.random() < 0.4
        if has_tests:
            src = next((f.path for f in files if f.path.endswith(".py")), "src/util/mod0.py")
            stem = src.rsplit("/", 1)[-1]
            files.append(FileChange(f"tests/test_{stem}", rng.randint(5, 50), 0, is_test=True))

        churn = sum(f.added + f.deleted for f in files)
        logit = (
            -3.0
            + 2.2 * touched_risky
            + 1.8 * missing_partner
            + 0.35 * math.log1p(churn)
            - 0.9 * has_tests
        )
        fail = rng.random() < 1 / (1 + math.exp(-logit))
        excluded = rng.random() < 0.08  # flaky/infra noise
        is_default = rng.random() < 0.3
        sha = _sha(rng)
        units.append(
            Unit(
                key=sha,
                as_of=t,
                completed_at=t + rng.uniform(600, 3600),
                label=None if excluded else int(fail),
                files=files,
                author=f"author{rng.randint(0, 9)}",
                is_pr=not is_default,
                is_default_branch=is_default,
                base_sha=default_shas[-rng.randint(1, min(5, len(default_shas)))]
                if default_shas and not is_default
                else None,
            )
        )
        if is_default:
            default_shas.append(sha)
    return units


def make_test_runs(n: int = 300, seed: int = 0):
    """Per-test history: tests for src/modK.py fail far more when src/modK.py changes."""
    from preflight_core.models.test_ranker import TestRun

    rng = random.Random(seed)
    tests = [f"tests/test_mod{i}.py::test_{j}" for i in range(15) for j in range(4)]
    runs = []
    for k in range(n):
        changed_mod = rng.randrange(15)
        results = {}
        for t in tests:
            p = 0.25 if f"test_mod{changed_mod}." in t else 0.005
            results[t] = ("failed" if rng.random() < p else "passed", rng.randint(10, 500))
        runs.append(
            TestRun(
                str(k),
                T0 + k * 3600.0,
                T0 + k * 3600.0 + 1800,
                [f"src/mod{changed_mod}.py"],
                results,
            )
        )
    return runs
