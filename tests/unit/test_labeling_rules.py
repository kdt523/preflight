import pytest

from preflight_core.labeling.rules import (
    CANCELLED,
    CODE,
    CONFIG,
    FLAKY,
    INFRA,
    PASS,
    JobEvidence,
    classify_build,
    classify_job,
)


def job(log=None, step=None, conclusion="failure"):
    return JobEvidence(conclusion, step, log.splitlines() if log else None)


LOGS = {
    "pytest_fail": (
        CODE,
        "pytest_failed",
        """
tests/test_api.py::test_login FAILED
________________________ test_login ________________________
<source redacted>
E       assert 500 == 200
tests/test_api.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_api.py::test_login - assert 500 == 200
========================= 1 failed, 212 passed in 31.20s =========================
Error: Process completed with exit code 1.""",
    ),
    "import_error": (
        CODE,
        "python_error",
        """
ImportError while loading conftest '/home/runner/work/app/conftest.py'.
ModuleNotFoundError: No module named 'app.utils'""",
    ),
    "mypy": (
        CODE,
        "type_check",
        """
src/app/api.py:42: error: Incompatible return value type (got "int", expected "str")  [return-value]
Found 1 error in 1 file (checked 88 source files)""",
    ),
    "ruff": (
        CODE,
        "lint",
        """
src/app/api.py:3:1: F401 [*] `os` imported but unused
Found 1 error.""",
    ),
    "runner_shutdown": (
        INFRA,
        "runner_shutdown",
        """
collected 3000 items
##[error]The runner has received a shutdown signal.""",
    ),
    "oom_beats_test_failure": (
        INFRA,
        "oom",
        """
FAILED tests/test_big.py::test_matrix
/home/runner/work/_temp/abc.sh: line 1:  1234 Killed
Error: Process completed with exit code 137.""",
    ),
    "disk": (INFRA, "disk_full", "OSError: [Errno 28] No space left on device"),
    "pip_conflict": (
        CONFIG,
        "pip_resolution",
        """
ERROR: Cannot install app==1.0 because these package versions have conflicting dependencies.
ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/...""",
    ),
    "network_only": (
        INFRA,
        "network",
        """
Downloading https://files.pythonhosted.org/packages/...
curl: (6) Could not resolve host: files.pythonhosted.org""",
    ),
    "network_exception": (
        INFRA,
        "network_exception",
        """
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.x.com', port=443)""",
    ),
    "network_inside_failing_test_is_code": (
        CODE,
        "pytest_failed",
        """
E   requests.exceptions.ConnectionError: HTTPSConnectionPool(host='localhost', port=8080)
FAILED tests/test_client.py::test_fetch - requests.exceptions.ConnectionError""",
    ),
    "rate_limit": (
        INFRA,
        "rate_limit",
        "Error: toomanyrequests: You have reached your pull rate limit.",
    ),
}


@pytest.mark.parametrize("name", LOGS)
def test_log_rules(name):
    want_cat, want_reason, log = LOGS[name]
    cat, source, reason = classify_job(job(log=log))
    assert (cat, source, reason) == (want_cat, "log", want_reason)


@pytest.mark.parametrize(
    "step, want",
    [
        ("Set up job", INFRA),
        ("Run actions/checkout@v4", INFRA),
        ("Restore cache", INFRA),
        ("Install dependencies", CONFIG),
        ("Set up Python 3.12", CONFIG),
        ("Run pytest", CODE),
        ("Lint with ruff", CODE),
        ("Type check", CODE),
        ("Upload coverage", None),
    ],
)
def test_step_fallback_when_log_expired(step, want):
    cat, source, _ = classify_job(job(step=step))
    assert cat == want
    assert source == ("step" if want else "none")


def test_log_beats_step_name():
    cat, source, _ = classify_job(job(log="No space left on device", step="Run pytest"))
    assert (cat, source) == (INFRA, "log")


def test_job_timeout_is_infra():
    assert classify_job(job(conclusion="timed_out"))[0] == INFRA


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped", "action_required", None])
def test_dropped_conclusions(conclusion):
    assert classify_build(conclusion, False, []).failure_type == CANCELLED


def test_success_is_pass():
    assert classify_build("success", False, []).failure_type == PASS


def test_later_success_is_flaky_even_with_code_evidence():
    label = classify_build("failure", True, [job(log=LOGS["pytest_fail"][2])])
    assert (label.failure_type, label.source) == (FLAKY, "rerun")


def test_run_level_conclusions():
    assert classify_build("timed_out", False, []).failure_type == INFRA
    assert classify_build("startup_failure", False, []).failure_type == CONFIG


def test_any_code_job_makes_build_code_defect():
    jobs = [job(log="Could not resolve host: pypi.org"), job(step="Run pytest")]
    label = classify_build("failure", False, jobs)
    assert (label.failure_type, label.source, label.reason) == (CODE, "step", "step:Run pytest")


def test_only_infra_jobs_make_build_infra():
    label = classify_build("failure", False, [job(step="Set up job"), job(conclusion="timed_out")])
    assert label.failure_type == INFRA


def test_successful_jobs_are_ignored():
    jobs = [job(step="Run pytest", conclusion="success"), job(step="Set up job")]
    assert classify_build("failure", False, jobs).failure_type == INFRA


def test_unexplained_failure_defaults_to_code_with_default_source():
    label = classify_build("failure", False, [job(step="Upload coverage")])
    assert (label.failure_type, label.source) == (CODE, "default")
