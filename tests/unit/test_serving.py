import json
import time

import fakeredis
import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from apps.scorer.app import create_app
from preflight_core.features.state import build_state, replay
from preflight_core.github.app_auth import InstallationTokens, app_jwt
from preflight_core.localgit import parse_numstat
from preflight_core.models.registry import BundleProvider, LocalRegistry
from preflight_core.online.store import OnlineStore
from preflight_core.serving.render import check_run_payload, skipped_payload, summary, title
from preflight_core.serving.service import ScoreRequest, ScoringService, ci_plan
from pytest_plugin.plugin import reorder


@pytest.fixture
def store():
    return OnlineStore(fakeredis.FakeRedis())


# -- train/serve parity ---------------------------------------------------------------


@pytest.mark.parametrize("i", [100, 900, 1999])
def test_online_state_gives_identical_features_to_offline_replay(synth_units, store, i):
    """SPEC §8 parity test: materialise -> Redis -> features == offline training features."""
    offline = {u.key: f for u, f, _ in replay(synth_units)}
    target = synth_units[i]
    state, started, completed = build_state(synth_units, until=target.as_of)
    store.replace_snapshot("acme/app", state, started, completed)
    online, _ = store.get_state("acme/app").features(target)
    for k, v in offline[target.key].items():
        assert online[k] == pytest.approx(v, nan_ok=True), k


def test_snapshot_records_applied_events(synth_units, store):
    state, started, completed = build_state(synth_units, until=synth_units[50].as_of)
    store.replace_snapshot("r", state, started, completed)
    assert store.was_applied("r", "start", synth_units[10].key)
    assert not store.was_applied("r", "start", synth_units[60].key)


# -- scoring service ------------------------------------------------------------------------


def req(**kw):
    base = dict(
        repo="acme/app",
        sha="a" * 40,
        files=[
            {"path": "src/core/engine.py", "added": 120, "deleted": 30},
            {"path": "src/api/views.py", "added": 40, "deleted": 5},
        ],
    )
    return ScoreRequest(**{**base, **kw})


def service(store, bundle, synth_units, shadow=None):
    state, started, completed = build_state(synth_units, until=time.time())
    store.replace_snapshot("acme/app", state, started, completed)
    return ScoringService(store, lambda: bundle, "salt", shadow=lambda: shadow)


def test_score_returns_everything_a_check_needs(store, synth_bundle, synth_units):
    out = service(store, synth_bundle, synth_units).score(req(author_login="dev@x.com"))
    assert 0 <= out["p_fail"] <= 1 and out["band"] in {"low", "watch", "high"}
    assert out["ci_plan"] in {"lint+affected", "full", "full-matrix"}
    assert out["model_version"] == "vsynth" and out["latency_ms"] > 0
    assert not any(k.startswith("author") for k in out["contributions"])
    assert "dev@x.com" not in json.dumps(out, default=str)  # raw identity never echoed


def test_scoring_latency_budget(store, synth_bundle, synth_units):
    svc = service(store, synth_bundle, synth_units)
    svc.score(req())  # warm-up
    lat = sorted(svc.score(req(sha=f"{i:040d}"))["latency_ms"] for i in range(40))
    assert lat[int(0.95 * len(lat)) - 1] < 300  # SPEC: p95 < 300 ms


def test_blame_free_repo_uses_model_without_author_features(store, synth_bundle, synth_units):
    svc = service(store, synth_bundle, synth_units)
    store.set_repo_config("acme/app", blame_free=True)
    assert svc.score(req())["model_key"] == "global-blame-free"


def test_ci_plan_rules():
    assert ci_plan("low", False, False) == "lint+affected"
    assert ci_plan("high", False, False) == "full-matrix"
    assert ci_plan("low", True, False) == "full"  # abstain -> run everything
    assert ci_plan("low", False, True) == "full-matrix"  # default branch: always full


def test_test_ordering_puts_related_tests_first(store, synth_bundle, synth_units):
    from tests.synth import make_test_runs

    svc = service(store, synth_bundle, synth_units)
    hist = store.get_tests("acme/app")
    for r in make_test_runs(200):
        hist.apply(r)
    store.put_tests("acme/app", hist)
    tests = [f"tests/test_mod{i}.py::test_0" for i in range(15)]
    order = svc.order_tests("acme/app", ["src/mod7.py"], tests)
    assert order[0] == "tests/test_mod7.py::test_0"
    assert sorted(order) == sorted(tests)


# -- registry -------------------------------------------------------------------------


def test_registry_promote_shadow_and_rollback(tmp_path, synth_bundle):
    reg = LocalRegistry(tmp_path)
    for v in ("v1", "v2"):
        synth_bundle.version = v
        synth_bundle.save(tmp_path / "bundles")
    synth_bundle.version = "vsynth"
    reg.promote("v1")
    reg.set_shadow("v2")
    prov = BundleProvider(reg, reload_s=0)
    assert (prov.champion().version, prov.shadow().version) == ("v1", "v2")
    reg.promote("v2")
    assert reg.read()["champion"] == "v2" and reg.read()["shadow"] is None
    assert reg.rollback() == "v1"
    prov.force_reload()
    assert prov.champion().version == "v1"
    with pytest.raises(RuntimeError):
        reg.rollback()


# -- scorer API -----------------------------------------------------------------------------


def test_scorer_api(store, synth_bundle, synth_units):
    client = TestClient(create_app(service(store, synth_bundle, synth_units)))
    body = {
        "repo": "acme/app",
        "sha": "b" * 40,
        "files": [{"path": "src/core/parser.py", "added": 10, "deleted": 2}],
    }
    r = client.post("/v1/score", json=body)
    assert r.status_code == 200 and "_features" not in r.json()
    assert client.get("/readyz").json()["champion"] == "vsynth"
    assert b"preflight_score_latency_seconds" in client.get("/metrics").content
    t = client.post(
        "/v1/tests/order", json={"repo": "nobody/else", "changed_files": [], "tests": ["b", "a"]}
    )
    assert t.json()["order"] and t.status_code == 200


def test_scorer_without_model_is_503(store):
    client = TestClient(create_app(ScoringService(store, lambda: None, "s")))
    r = client.post("/v1/score", json={"repo": "a/b", "sha": "c" * 40, "files": []})
    assert r.status_code == 503
    assert client.get("/readyz").status_code == 503


# -- check run rendering -----------------------------------------------------------------------

SCORE = {
    "p_fail": 0.71,
    "band": "high",
    "abstain": False,
    "ci_plan": "full-matrix",
    "model_version": "v1",
    "model_key": "global",
    "latency_ms": 42.0,
    "reasons": [{"direction": "raises", "text": "`auth/session.py` failed 16 of 42 builds"}],
    "what_ifs": [{"text": "Add tests", "p_before": 0.71, "p_after": 0.52}],
    "test_order": ["tests/test_auth.py::test_login"],
}


def test_check_run_never_blocks():
    p = check_run_payload("a" * 40, SCORE)
    assert p["conclusion"] == "neutral" and p["status"] == "completed"
    assert {a["identifier"] for a in p["actions"]} == {"useful", "not_useful"}
    assert title(SCORE).startswith("🔴 High risk · 71%")
    s = summary(SCORE)
    assert "71% → **52%**" in s and "`full-matrix`" in s and "test_login" in s
    sk = skipped_payload("a" * 40, "scorer timeout")
    assert sk["conclusion"] == "skipped" and "CI runs normally" in sk["output"]["title"]
    assert title({**SCORE, "abstain": True}).startswith("⚪ Not enough signal")


# -- GitHub App auth ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def keypair():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    return pem, k.public_key()


def test_app_jwt_is_short_lived_and_signed(keypair):
    pem, pub = keypair
    claims = jwt.decode(
        app_jwt("123", pem, now=1_000_000),
        pub,
        algorithms=["RS256"],
        options={"verify_exp": False, "verify_iat": False},
    )
    assert claims["iss"] == "123" and claims["exp"] - claims["iat"] <= 600


@respx.mock
def test_installation_tokens_are_cached_until_near_expiry(keypair):
    pem, _ = keypair
    clock = [1_800_000_000.0]
    exp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(clock[0] + 3600))
    route = respx.post("https://api.github.com/app/installations/9/access_tokens").mock(
        return_value=httpx.Response(201, json={"token": "ghs_x", "expires_at": exp})
    )
    t = InstallationTokens("123", pem, clock=lambda: clock[0])
    assert t.token(9) == t.token(9) == "ghs_x" and route.call_count == 1
    clock[0] += 3600 - 200  # inside the 5-minute refresh margin
    t.token(9)
    assert route.call_count == 2


# -- local git + pytest plugin ---------------------------------------------------------------


def test_parse_numstat_handles_renames_and_binaries():
    out = "10\t2\tsrc/a.py\n-\t-\tlogo.png\n3\t1\tsrc/{old => new}/b.py\n1\t0\tx.py => y.py"
    files = {f["path"]: f for f in parse_numstat(out)}
    assert set(files) == {"src/a.py", "logo.png", "src/new/b.py", "y.py"}
    assert files["logo.png"]["added"] == 0 and files["src/a.py"]["added"] == 10


def test_plugin_reorder_keeps_unknown_items_in_place():
    class It:
        def __init__(self, nid):
            self.nodeid = nid

    items = [It(n) for n in ("a", "b", "c", "d")]
    assert [i.nodeid for i in reorder(items, ["c", "a"])] == ["c", "a", "b", "d"]


def test_scorer_requires_bearer_token_when_configured(store, synth_bundle, synth_units):
    client = TestClient(create_app(service(store, synth_bundle, synth_units), api_token="t0k"))
    body = {"repo": "acme/app", "sha": "b" * 40, "files": []}
    assert client.post("/v1/score", json=body).status_code == 401
    bad = client.post("/v1/score", json=body, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    ok = client.post("/v1/score", json=body, headers={"Authorization": "Bearer t0k"})
    assert ok.status_code == 200
    assert client.get("/healthz").status_code == 200  # probes stay open


def test_settings_build_database_url_from_rds_parts(monkeypatch):
    from preflight_core.config import Settings

    monkeypatch.setenv("DB_HOST", "db.internal")
    monkeypatch.setenv("DB_PASSWORD", "p@ss/w:rd")
    monkeypatch.setenv("S3_ENDPOINT_URL", "")
    s = Settings(_env_file=None)
    assert (
        s.database_url
        == "postgresql+psycopg://preflight:p%40ss%2Fw%3Ard@db.internal:5432/preflight"
    )
    assert s.s3_endpoint_url is None
