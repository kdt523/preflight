"""Postgres schema (SPEC §6). Week 1 tables only; later weeks add features, predictions, etc.

No source code is ever stored: file_changes holds paths + line counts; authors are hashed.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

repos = Table(
    "repos",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("full_name", String(200), nullable=False, unique=True),
    Column("installation_id", BigInteger),
    Column("default_branch", String(200), nullable=False),
    Column("failure_base_rate", Float),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

commits = Table(
    "commits",
    metadata,
    Column("repo_id", Integer, ForeignKey("repos.id"), nullable=False),
    Column("sha", String(40), nullable=False),
    Column("author_hash", String(64)),
    Column("committed_at", DateTime(timezone=True), nullable=False),
    Column("parent_sha", String(40)),
    Column("parent_count", Integer, nullable=False, server_default="1"),
    Column("additions", Integer, nullable=False),
    Column("deletions", Integer, nullable=False),
    Column("files_changed", Integer, nullable=False),
    # GitHub truncates commit file lists at 300 files; features must know the list is partial.
    Column("files_truncated", Boolean, nullable=False, server_default="false"),
    PrimaryKeyConstraint("repo_id", "sha"),
)

file_changes = Table(
    "file_changes",
    metadata,
    Column("repo_id", Integer, nullable=False),
    Column("sha", String(40), nullable=False),
    Column("path", Text, nullable=False),
    Column("previous_path", Text),
    Column("added", Integer, nullable=False),
    Column("deleted", Integer, nullable=False),
    Column("change_type", String(20), nullable=False),  # added|modified|removed|renamed|...
    Column("is_test", Boolean, nullable=False),
    PrimaryKeyConstraint("repo_id", "sha", "path"),
)

# One build = one attempt of one workflow run. Reruns are new rows linked via rerun_of.
builds = Table(
    "builds",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("repo_id", Integer, ForeignKey("repos.id"), nullable=False),
    Column("run_id", BigInteger, nullable=False),
    Column("run_attempt", Integer, nullable=False),
    Column("workflow_id", BigInteger, nullable=False),
    Column("workflow_name", Text),
    Column("sha", String(40), nullable=False),
    Column("head_branch", Text),
    Column("event", String(50), nullable=False),
    Column("pr_number", Integer),
    Column("base_sha", String(40)),  # PR base at run time; lets week 2 rebuild the PR diff as-of
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("conclusion", String(30)),  # GitHub conclusion: success|failure|cancelled|...
    Column("failure_type", String(20)),  # pass|code_defect|flaky|infra|config|cancelled
    Column("label_source", String(20)),  # rerun|log|step|conclusion|default
    Column("label_reason", Text),
    Column("rerun_of", BigInteger, ForeignKey("builds.id")),
    Column("log_available", Boolean),
    UniqueConstraint("repo_id", "run_id", "run_attempt", name="uq_builds_run_attempt"),
    Index("ix_builds_repo_sha", "repo_id", "sha"),
    Index("ix_builds_repo_started", "repo_id", "started_at"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=False),  # GitHub job id
    Column("build_id", BigInteger, ForeignKey("builds.id"), nullable=False, index=True),
    Column("name", Text, nullable=False),
    Column("conclusion", String(30)),
    Column("failed_step", Text),
    Column("runner_name", Text),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("log_tail_key", Text),  # S3 key of the last 200 log lines, if fetched
)

test_results = Table(
    "test_results",
    metadata,
    Column("build_id", BigInteger, ForeignKey("builds.id"), nullable=False),
    Column("test_id", Text, nullable=False),
    Column("status", String(20), nullable=False),  # passed|failed|error|skipped
    Column("duration_ms", Integer),
    Column("flaky_score", Float),
    PrimaryKeyConstraint("build_id", "test_id"),
)

webhook_deliveries = Table(
    "webhook_deliveries",
    metadata,
    Column("delivery_id", String(64), primary_key=True),
    Column("event", String(50), nullable=False),
    Column("action", String(50)),
    Column("repo_full_name", String(200)),
    Column("received_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

backfill_state = Table(
    "backfill_state",
    metadata,
    Column("repo_id", Integer, ForeignKey("repos.id"), nullable=False),
    Column("stage", String(40), nullable=False),
    Column("cursor", JSONB, nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    PrimaryKeyConstraint("repo_id", "stage"),
)

http_cache = Table(
    "http_cache",
    metadata,
    Column("url", Text, primary_key=True),
    Column("etag", Text, nullable=False),
    Column("body", LargeBinary, nullable=False),  # gzipped JSON
    Column("fetched_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

# Point-in-time feature vectors (SPEC §6). `details` holds explanation context (e.g. the
# riskiest file and its counts); it is never a model input.
features = Table(
    "features",
    metadata,
    Column("repo_id", Integer, ForeignKey("repos.id"), nullable=False),
    Column("sha", String(40), nullable=False),
    Column("as_of", DateTime(timezone=True), nullable=False),
    Column("feature_set_version", String(20), nullable=False),
    Column("vector", JSONB, nullable=False),
    Column("details", JSONB),
    PrimaryKeyConstraint("repo_id", "sha", "feature_set_version"),
)

# SPEC §6: one row per trained bundle; status: candidate -> shadow -> champion -> retired
model_runs = Table(
    "model_runs",
    metadata,
    Column("version", String(40), primary_key=True),
    Column("data_window", Text, nullable=False),
    Column("feature_hash", String(32), nullable=False),
    Column("pr_auc", Float),
    Column("ece", Float),
    Column("threshold", Float),
    Column("status", String(20), nullable=False, server_default="candidate"),
    Column("metrics", JSONB),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("repo_id", Integer, ForeignKey("repos.id"), nullable=False),
    Column("sha", String(40), nullable=False),
    Column("pr_number", Integer),
    Column("model_version", String(40), nullable=False),
    Column("model_key", String(200), nullable=False),  # global | repo:<name> | global-blame-free
    Column("shadow", Boolean, nullable=False, server_default="false"),
    Column("p_fail", Float),
    Column("band", String(10)),  # low|watch|high|skipped
    Column("conformal_set", JSONB),
    Column("shap_top", JSONB),  # visible reasons only; author features never stored here
    Column("what_ifs", JSONB),
    Column("ci_plan", String(20)),
    Column("latency_ms", Float),
    Column("fail_open_reason", Text),
    Column("check_run_id", BigInteger),
    Column("features", JSONB),  # served vector, for drift monitoring and parity audits
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Index("ix_predictions_repo_sha", "repo_id", "sha"),
    Index("ix_predictions_created", "created_at"),
)

feedback = Table(
    "feedback",
    metadata,
    Column("prediction_id", BigInteger, ForeignKey("predictions.id"), nullable=False),
    Column("user_hash", String(64), nullable=False),
    Column("useful", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    PrimaryKeyConstraint("prediction_id", "user_hash"),
)
