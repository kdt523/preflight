"""row level security

Tenant isolation by GitHub App installation (SPEC §6). The pipeline role owns the tables and
is not subject to RLS; tenant-facing reads run as `preflight_tenant` with
`app.installation_id` set per transaction (see preflight_core/db/tenancy.py). An unset
setting matches nothing, so a forgotten SET fails closed.

Revision ID: 4bf46f1c64a6
Revises: 02ec6e4d65db
"""

from alembic import op

revision = "4bf46f1c64a6"
down_revision = "02ec6e4d65db"
branch_labels = None
depends_on = None

TENANT = "preflight_tenant"
INSTALLATION = "NULLIF(current_setting('app.installation_id', true), '')::bigint"
REPO_SCOPED = ["builds", "commits", "file_changes", "features", "predictions"]


def upgrade() -> None:
    op.execute(f"""
        DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{TENANT}') THEN
            CREATE ROLE {TENANT} NOLOGIN;
          END IF;
        END $$;
    """)
    op.execute(f"GRANT {TENANT} TO CURRENT_USER")
    tables = ["repos", *REPO_SCOPED, "jobs", "test_results", "feedback"]
    op.execute(f"GRANT SELECT ON {', '.join(tables)} TO {TENANT}")

    op.execute("ALTER TABLE repos ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY tenant_isolation ON repos TO {TENANT} "
               f"USING (installation_id = {INSTALLATION})")
    tenant_repos = f"SELECT id FROM repos WHERE installation_id = {INSTALLATION}"
    for t in REPO_SCOPED:
        op.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY tenant_isolation ON {t} TO {TENANT} "
                   f"USING (repo_id IN ({tenant_repos}))")
    for t in ("jobs", "test_results"):
        op.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY tenant_isolation ON {t} TO {TENANT} USING (build_id IN "
                   f"(SELECT id FROM builds WHERE repo_id IN ({tenant_repos})))")
    op.execute("ALTER TABLE feedback ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY tenant_isolation ON feedback TO {TENANT} USING (prediction_id IN "
               f"(SELECT id FROM predictions WHERE repo_id IN ({tenant_repos})))")


def downgrade() -> None:
    for t in ["repos", *REPO_SCOPED, "jobs", "test_results", "feedback"]:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {t}")
        op.execute(f"ALTER TABLE {t} DISABLE ROW LEVEL SECURITY")
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {TENANT}")
