"""Row-level security and live shadow-vs-champion evaluation against real Postgres."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, text

from preflight_core.db.schema import builds, predictions, repos
from preflight_core.db.tenancy import tenant_connection
from preflight_core.models.promotion import compare_live, labelled_predictions

pytestmark = pytest.mark.integration

T0 = datetime.now(UTC) - timedelta(days=10)


def seed(db, n=60):
    with db.begin() as c:
        c.execute(
            insert(repos),
            [
                {"id": 1, "full_name": "a/one", "default_branch": "main", "installation_id": 111},
                {"id": 2, "full_name": "b/two", "default_branch": "main", "installation_id": 222},
            ],
        )
        rows_b, rows_p = [], []
        for i in range(n):
            sha = f"{i:040d}"
            fail = i % 3 == 0
            rows_b.append(
                {
                    "repo_id": 1,
                    "run_id": i,
                    "run_attempt": 1,
                    "workflow_id": 1,
                    "sha": sha,
                    "event": "push",
                    "started_at": T0 + timedelta(hours=i),
                    "conclusion": "failure" if fail else "success",
                    "failure_type": "code_defect" if fail else "pass",
                }
            )
            # champion is informative; shadow is even better
            rows_p.append(
                {
                    "repo_id": 1,
                    "sha": sha,
                    "model_version": "champ",
                    "model_key": "g",
                    "shadow": False,
                    # half the failures score below the passes: informative, not perfect
                    "p_fail": (0.6 if i % 2 else 0.2) if fail else 0.3,
                    "created_at": T0 + timedelta(hours=i),
                }
            )
            rows_p.append(
                {
                    "repo_id": 1,
                    "sha": sha,
                    "model_version": "shad",
                    "model_key": "g",
                    "shadow": True,
                    "p_fail": 0.9 if fail else 0.1,
                    "created_at": T0 + timedelta(hours=i),
                }
            )
        rows_b.append(
            {
                "repo_id": 2,
                "run_id": 999,
                "run_attempt": 1,
                "workflow_id": 1,
                "sha": "f" * 40,
                "event": "push",
                "started_at": T0,
                "conclusion": "success",
                "failure_type": "pass",
            }
        )
        c.execute(insert(builds), rows_b)
        c.execute(insert(predictions), rows_p)


def test_rls_isolates_installations(db):
    seed(db)
    with tenant_connection(db, 111) as c:
        assert c.execute(text("SELECT full_name FROM repos")).scalars().all() == ["a/one"]
        assert c.execute(text("SELECT count(*) FROM builds WHERE repo_id = 2")).scalar() == 0
        assert c.execute(text("SELECT count(*) FROM predictions")).scalar() == 120
    with tenant_connection(db, 222) as c:
        assert c.execute(text("SELECT count(*) FROM builds")).scalar() == 1
        assert c.execute(text("SELECT count(*) FROM predictions")).scalar() == 0
    with db.connect() as c:  # the pipeline role (table owner) still sees everything
        assert c.execute(text("SELECT count(*) FROM repos")).scalar() == 2


def test_rls_fails_closed_without_installation(db):
    seed(db)
    with db.begin() as c:
        c.execute(text("SET LOCAL ROLE preflight_tenant"))
        assert c.execute(text("SELECT count(*) FROM repos")).scalar() == 0


def test_live_comparison_prefers_the_better_shadow(db):
    seed(db)
    df = labelled_predictions(db, T0 - timedelta(days=1))
    assert set(df["y"]) == {0, 1} and len(df) == 120
    verdict = compare_live(db, "champ", "shad", days=30)
    assert verdict["shadow"]["pr_auc"] > verdict["champion"]["pr_auc"]
    assert verdict["promote"], verdict["reason"]
