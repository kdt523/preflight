"""Dashboard data layer against real Postgres (seeded like test_ops) + an AppTest smoke run."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import insert

from apps.dashboard import data
from preflight_core.db.schema import feedback, file_changes
from preflight_core.db.tenancy import tenant_connection
from tests.integration.test_ops import seed

pytestmark = pytest.mark.integration
SINCE = datetime.now(UTC) - timedelta(days=30)


def test_kpis_and_estimates(db):
    seed(db)
    with db.begin() as c:
        c.execute(
            insert(feedback),
            [
                {"prediction_id": 1, "user_hash": "u1", "useful": True},
                {"prediction_id": 3, "user_hash": "u1", "useful": False},
            ],
        )
    with db.connect() as c:
        chk, out = data.checks(c, SINCE), data.outcomes(c, SINCE)
    k = data.kpis(chk, out)
    assert k["checks"] == 60 and k["labelled"] == 60 and k["useful_share"] == 0.5
    assert 0 < k["live_pr_auc"] <= 1 and k["fail_open_rate"] == 0
    est = data.ci_minutes(out.assign(band="low", minutes=10.0), 0.5)
    assert est["estimated_saved"] == pytest.approx(300.0) and est["low_risk_failures"] == 20


def test_hotspots_rank_churn_and_smooth_rates(db):
    seed(db)
    with db.begin() as c:
        c.execute(
            insert(file_changes),
            [
                {
                    "repo_id": 1,
                    "sha": f"{i:040d}",
                    "path": "src/core/engine.py",
                    "added": 50,
                    "deleted": 10,
                    "change_type": "modified",
                    "is_test": False,
                }
                for i in range(0, 30)
            ]
            + [
                {
                    "repo_id": 1,
                    "sha": f"{i:040d}",
                    "path": "README.md",
                    "added": 1,
                    "deleted": 0,
                    "change_type": "modified",
                    "is_test": False,
                }
                for i in range(30, 60)
            ],
        )
    with db.connect() as c:
        h = data.hotspots(c, 1).set_index("path")
    assert h.loc["src/core/engine.py", "churn"] == 30 * 60
    assert (
        h.loc["src/core/engine.py", "dir"] == "src/core" and h.loc["README.md", "dir"] == "(root)"
    )
    assert 0 < h.loc["README.md", "fail_rate"] < 1  # smoothed, never a raw 0 or 1


def test_dashboard_data_respects_tenancy(db):
    seed(db)
    with tenant_connection(db, 222) as c:
        assert len(data.checks(c, SINCE)) == 0 and data.repos(c)["full_name"].tolist() == ["b/two"]


def test_app_renders_without_errors(db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import preflight_core.db.engine as eng

    seed(db)
    eng.get_engine.cache_clear()
    monkeypatch.setattr(eng, "get_engine", lambda: db)
    app_py = Path(__file__).parents[2] / "apps" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_py), default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Hotspots", "Calibration", "Drift"]
    assert at.metric[0].value == "60"
