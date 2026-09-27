"""Runs pytest-in-pytest with the Preflight plugin active against a mocked scorer."""

import httpx
import respx

pytest_plugins = ["pytester"]

TESTS = """
def test_a(): pass
def test_b(): pass
def test_c(): pass
"""
ARGS = [
    "--preflight",
    "--preflight-repo",
    "acme/app",
    "--preflight-url",
    "http://scorer",
    "-v",
    "-p",
    "no:randomly",
]


def _order(result):
    return [
        ln.split("::")[1].split()[0]
        for ln in result.outlines
        if "::test_" in ln and ("PASSED" in ln or "FAILED" in ln)
    ]


@respx.mock
def test_plugin_reorders_by_server_ranking(pytester):
    pytester.makepyfile(test_mod=TESTS)
    route = respx.post("http://scorer/v1/tests/order").mock(
        side_effect=lambda req: httpx.Response(
            200,
            json={
                "order": ["test_mod.py::test_c", "test_mod.py::test_a", "test_mod.py::test_b"],
                "reordered": True,
            },
        )
    )
    result = pytester.runpytest_inprocess(*ARGS)
    assert _order(result) == ["test_c", "test_a", "test_b"]
    assert route.called
    sent = route.calls.last.request.read()
    assert b"test_mod.py::test_a" in sent and b"acme/app" in sent
    result.stdout.fnmatch_lines(["*preflight: reordered 3 tests*"])


@respx.mock
def test_plugin_fails_open_on_server_error(pytester):
    pytester.makepyfile(test_mod=TESTS)
    respx.post("http://scorer/v1/tests/order").mock(return_value=httpx.Response(500))
    result = pytester.runpytest_inprocess(*ARGS)
    assert _order(result) == ["test_a", "test_b", "test_c"]
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(["*preflight: ordering skipped (HTTPStatusError)*"])


def test_plugin_is_inert_without_flag(pytester):
    pytester.makepyfile(test_mod=TESTS)
    result = pytester.runpytest_inprocess("-v")
    assert _order(result) == ["test_a", "test_b", "test_c"]
    assert "preflight:" not in result.stdout.str()
