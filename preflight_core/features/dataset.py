"""Stored builds -> prediction units -> point-in-time feature table.

A unit is one commit's *first* CI outcome in a repo (attempt 1 of every CI workflow run on
that sha): as_of = earliest start, label known at the latest completion.
  label 1  any workflow failed as code_defect
  label 0  every workflow passed
  None     only flaky / infra / config failures (excluded from training, still replayed)
"""

import io
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import insert

from preflight_core.db.schema import features as features_table
from preflight_core.features.definitions import FEATURE_SET_VERSION, FEATURES, feature_names
from preflight_core.features.diff import FileChange
from preflight_core.features.state import Unit, replay

_BUILDS = text(
    """
    SELECT sha, started_at, completed_at, failure_type, event, head_branch, base_sha
    FROM builds WHERE repo_id = :repo AND run_attempt = 1
      AND (CAST(:sha AS text) IS NULL OR sha = :sha)
    """
)
_FILES = text(
    "SELECT sha, path, added, deleted, change_type, is_test FROM file_changes "
    "WHERE repo_id = :repo AND (CAST(:sha AS text) IS NULL OR sha = :sha)"
)
_COMMITS = text(
    "SELECT sha, author_hash, files_truncated FROM commits "
    "WHERE repo_id = :repo AND (CAST(:sha AS text) IS NULL OR sha = :sha)"
)


def unit_label(failure_types: list[str]) -> int | None:
    types = [t for t in failure_types if t and t != "cancelled"]
    if "code_defect" in types:
        return 1
    if types and all(t == "pass" for t in types):
        return 0
    return None


def load_units(
    engine: Engine, repo_id: int, default_branch: str, sha: str | None = None
) -> tuple[list[Unit], set[str]]:
    """-> (units, shas that have a stored diff). `sha` restricts to one commit."""
    with engine.connect() as c:
        args = {"repo": repo_id, "sha": sha}
        builds = c.execute(_BUILDS, args).all()
        file_rows = c.execute(_FILES, args).all()
        commit_rows = {r.sha: r for r in c.execute(_COMMITS, args)}

    files: dict[str, list[FileChange]] = defaultdict(list)
    for f in file_rows:
        files[f.sha].append(FileChange(f.path, f.added, f.deleted, f.change_type, f.is_test))

    by_sha: dict[str, list[Any]] = defaultdict(list)
    for b in builds:
        by_sha[b.sha].append(b)

    units = []
    for sha, bs in by_sha.items():
        live = [b for b in bs if b.failure_type != "cancelled"] or bs
        done = [b.completed_at for b in live if b.completed_at] or [
            b.completed_at for b in bs if b.completed_at
        ]
        commit = commit_rows.get(sha)
        units.append(
            Unit(
                key=sha,
                as_of=min(b.started_at for b in bs).timestamp(),
                completed_at=max(done).timestamp() if done else None,
                label=unit_label([b.failure_type for b in bs]),
                files=files.get(sha, []),
                truncated=bool(commit and commit.files_truncated),
                author=commit.author_hash if commit else None,
                is_pr=any(b.event == "pull_request" for b in bs),
                is_default_branch=any(
                    b.event == "push" and b.head_branch == default_branch for b in bs
                ),
                base_sha=next((b.base_sha for b in bs if b.base_sha), None),
            )
        )
    return units, set(commit_rows)


def build_frame(units: list[Unit], with_diff: set[str], repo_id: int, repo: str) -> pd.DataFrame:
    rows = []
    for u, feats, details in replay(units):
        if u.key not in with_diff:
            continue  # no diff -> nothing to predict from; the unit still shaped history
        rows.append(
            {
                "repo_id": repo_id,
                "repo": repo,
                "sha": u.key,
                "as_of": datetime.fromtimestamp(u.as_of, UTC),
                "completed_at": datetime.fromtimestamp(u.completed_at, UTC)
                if u.completed_at is not None
                else pd.NaT,
                "label": u.label,
                **feats,
                "details": json.dumps(details, default=str),
            }
        )
    cols = ["repo_id", "repo", "sha", "as_of", "completed_at", "label", *feature_names(), "details"]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = df["label"].astype("Int8")
    for n in feature_names():
        df[n] = df[n].astype("float64")
    return df


def _is_utc(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC"


def _rate_check(col: str) -> pa.Column:
    return pa.Column(float, pa.Check.in_range(0, 1), nullable=False, name=col)


FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "repo_id": pa.Column(int),
        "sha": pa.Column(str, pa.Check.str_length(40, 40)),
        "as_of": pa.Column(checks=pa.Check(_is_utc, element_wise=False, error="as_of not UTC")),
        "label": pa.Column("Int8", pa.Check.isin([0, 1]), nullable=True),
        **{
            f.name: pa.Column(
                float,
                pa.Check.ge(0),
                nullable=f.name in {"pushes_since_base", "hours_since_last_green"},
            )
            for f in FEATURES
        },
        **{
            c: _rate_check(c)
            for c in (
                "max_file_fail_rate",
                "mean_file_fail_rate",
                "change_entropy",
                "frac_test_files",
                "frac_files_unseen",
                "max_missing_confidence",
                "frac_src_with_tests",
                "repo_recent_fail_rate",
                "author_file_familiarity",
                "author_fail_rate",
            )
        },
    },
    checks=[
        pa.Check(
            lambda d: (d["completed_at"].isna() | (d["completed_at"] >= d["as_of"])).all(),
            error="label known before the build started",
        ),
    ],
    unique=["repo_id", "sha"],
    strict=False,
    coerce=False,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    return FEATURE_SCHEMA.validate(df, lazy=True)


def save_features(engine: Engine, df: pd.DataFrame) -> None:
    names = feature_names()
    rows = [
        {
            "repo_id": int(r.repo_id),
            "sha": r.sha,
            "as_of": r.as_of.to_pydatetime(),
            "feature_set_version": FEATURE_SET_VERSION,
            "vector": {
                n: (None if math.isnan(v) else v)
                for n, v in zip(names, (getattr(r, n) for n in names), strict=True)
            },
            "details": json.loads(r.details),
        }
        for r in df.itertuples(index=False)
    ]
    if not rows:
        return
    with engine.begin() as c:
        for i in range(0, len(rows), 1000):
            stmt = insert(features_table).values(rows[i : i + 1000])
            stmt = stmt.on_conflict_do_update(
                index_elements=["repo_id", "sha", "feature_set_version"],
                set_={
                    "as_of": stmt.excluded.as_of,
                    "vector": stmt.excluded.vector,
                    "details": stmt.excluded.details,
                },
            )
            c.execute(stmt)


def dataset_path(root: str = "data/datasets") -> Path:
    return Path(root) / f"features_{FEATURE_SET_VERSION}.parquet"


def write_parquet(df: pd.DataFrame, path: Path, raw_store: Any | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if raw_store is not None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        raw_store.client.put_object(
            Bucket=raw_store.bucket, Key=f"datasets/{path.name}", Body=buf.getvalue()
        )


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or dataset_path())
