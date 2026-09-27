import subprocess

import httpx
import pytest
import respx

from cli.main import main, render

SCORE = {
    "p_fail": 0.64,
    "band": "high",
    "abstain": False,
    "ci_plan": "full-matrix",
    "model_version": "v1",
    "latency_ms": 38.0,
    "reasons": [{"direction": "raises", "text": "`src/auth.py` failed 9 of 20 builds"}],
    "what_ifs": [{"text": "Add or update tests", "p_before": 0.64, "p_after": 0.41}],
    "test_order": ["tests/test_auth.py::test_login"],
    "contributions": {},
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "dev@example.com")
    git("config", "user.name", "Dev")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n")  # uncommitted edit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PREFLIGHT_URL", "http://scorer")
    monkeypatch.setenv("PREFLIGHT_REPO", "acme/app")
    return tmp_path


@respx.mock
def test_check_sends_stats_only_and_prints_reasons(repo, capsys):
    route = respx.post("http://scorer/v1/score").mock(return_value=httpx.Response(200, json=SCORE))
    with pytest.raises(SystemExit) as e:
        main(["check"])
    assert e.value.code == 0  # never blocks by default
    sent = route.calls.last.request.read().decode()
    assert '"path":"a.py"' in sent.replace(" ", "") and '"added":2' in sent.replace(" ", "")
    assert "y = 2" not in sent  # no source code leaves the machine
    out = capsys.readouterr().out
    assert "🔴 HIGH risk: 64%" in out and "64% → 41%" in out and "test_login" in out


@respx.mock
def test_strict_mode_is_opt_in(repo):
    respx.post("http://scorer/v1/score").mock(return_value=httpx.Response(200, json=SCORE))
    with pytest.raises(SystemExit) as e:
        main(["check", "--strict"])
    assert e.value.code == 1


@respx.mock
def test_unreachable_scorer_fails_open(repo, capsys):
    respx.post("http://scorer/v1/score").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(SystemExit) as e:
        main(["check"])
    assert e.value.code == 0 and "push proceeds" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["plan"])
    assert capsys.readouterr().out.strip() == "full"


def test_install_hook(repo):
    with pytest.raises(SystemExit) as e:
        main(["install-hook"])
    assert e.value.code == 0
    assert "preflight check" in (repo / ".git/hooks/pre-push").read_text()


def test_render_abstain():
    assert render({**SCORE, "abstain": True, "_files": 1}).startswith("⚪ Not enough signal")
