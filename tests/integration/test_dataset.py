import pytest
from sqlalchemy import text

from preflight_core.features.dataset import build_frame, load_units, save_features, validate
from preflight_core.labeling.labeler import relabel_all
from tests.integration.test_backfill import (
    REPO,
    SHA_A,
    SHA_B,
    SHA_C,
    FakeGitHub,
    MemoryRawStore,
    make,
)

pytestmark = pytest.mark.integration


def test_units_features_and_storage_from_real_tables(db):
    raw = MemoryRawStore()
    make(db, FakeGitHub(), raw).run_repo(REPO, "main", junit=True)
    relabel_all(db, raw)
    with db.connect() as c:
        repo_id = c.execute(text("SELECT id FROM repos")).scalar()

    units, with_diff = load_units(db, repo_id, "main")
    by_key = {u.key: u for u in units}
    # A: attempt 1 was flaky -> excluded; B: infra -> excluded; C: pass + code_defect -> 1
    assert (by_key[SHA_A].label, by_key[SHA_B].label, by_key[SHA_C].label) == (None, None, 1)
    assert by_key[SHA_A].is_pr and not by_key[SHA_B].is_pr  # B was a push (to branch "x")
    assert with_diff == {SHA_A, SHA_C}  # B's commit was unreachable

    df = build_frame(units, with_diff, repo_id, REPO)
    validate(df)
    assert set(df["sha"]) == {SHA_A, SHA_C}
    save_features(db, df)
    save_features(db, df)  # idempotent
    with db.connect() as c:
        rows = c.execute(
            text(
                "SELECT sha, vector->>'n_files', details->'riskiest_file'->>'path' "
                "FROM features ORDER BY sha"
            )
        ).all()
    assert rows == [(SHA_A, "1.0", "src/auth.py"), (SHA_C, "1.0", "src/auth.py")]
