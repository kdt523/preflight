"""TravisTorrent (Beller, Gousios, Zaidman, MSR 2017) as an external benchmark.

The CSV has one row per *job*; we collapse to one row per build. Travis statuses map onto
our taxonomy: `failed` with failing tests -> code_defect (1), `passed` -> 0, `errored`
(Travis' infra/config status) -> excluded, `canceled` -> dropped.

Its feature set differs from Preflight's (no per-file history), so it is evaluated with its
own feature list under the same rolling-origin CV. Use it to sanity-check the pipeline and
to compare against published numbers, not to pretrain on mismatched features.
"""

import numpy as np
import pandas as pd

TT_FEATURES = [
    "n_files",
    "src_churn",
    "test_churn",
    "test_code_ratio",
    "files_added",
    "files_deleted",
    "files_modified",
    "tests_added",
    "src_files",
    "doc_files",
    "other_files",
    "commits_in_push",
    "team_size",
    "is_pr",
    "commits_on_files_touched",
    "sloc",
    "test_lines_per_kloc",
    "by_core_member",
    "prev_build_failed",
]

_USECOLS = [
    "tr_build_id",
    "gh_project_name",
    "gh_is_pr",
    "gh_build_started_at",
    "tr_status",
    "tr_duration",
    "tr_log_bool_tests_failed",
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_diff_tests_added",
    "gh_diff_src_files",
    "gh_diff_doc_files",
    "gh_diff_other_files",
    "gh_num_commits_in_push",
    "gh_team_size",
    "gh_num_commits_on_files_touched",
    "gh_sloc",
    "gh_test_lines_per_kloc",
    "gh_by_core_team_member",
    "git_trigger_commit",
]


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "t"]).astype(float)


def load(path: str, projects: list[str] | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=_USECOLS, chunksize=chunksize, low_memory=False):
        if projects:
            chunk = chunk[chunk["gh_project_name"].isin(projects)]
        parts.append(chunk.drop_duplicates("tr_build_id"))
    raw = pd.concat(parts, ignore_index=True).drop_duplicates("tr_build_id")
    return to_frame(raw)


def to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    status = raw["tr_status"].astype(str).str.lower()
    tests_failed = _bool(raw["tr_log_bool_tests_failed"]) > 0
    label = pd.Series(pd.NA, index=raw.index, dtype="Int8")
    label[status == "passed"] = 0
    label[(status == "failed") & tests_failed] = 1
    # failed without failing tests (compile/lint) is still a code failure in Travis' model
    label[(status == "failed") & ~tests_failed] = 1
    keep = status != "canceled"

    started = pd.to_datetime(raw["gh_build_started_at"], utc=True, errors="coerce")
    duration = pd.to_timedelta(pd.to_numeric(raw["tr_duration"], errors="coerce").fillna(0), "s")
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")  # noqa: E731
    df = pd.DataFrame(
        {
            "repo": raw["gh_project_name"],
            "sha": raw["git_trigger_commit"].astype(str),
            "build_id": raw["tr_build_id"],
            "as_of": started,
            "completed_at": started + duration,
            "label": label,
            "src_churn": num("git_diff_src_churn"),
            "test_churn": num("git_diff_test_churn"),
            "files_added": num("gh_diff_files_added"),
            "files_deleted": num("gh_diff_files_deleted"),
            "files_modified": num("gh_diff_files_modified"),
            "tests_added": num("gh_diff_tests_added"),
            "src_files": num("gh_diff_src_files"),
            "doc_files": num("gh_diff_doc_files"),
            "other_files": num("gh_diff_other_files"),
            "commits_in_push": num("gh_num_commits_in_push"),
            "team_size": num("gh_team_size"),
            "is_pr": _bool(raw["gh_is_pr"]),
            "commits_on_files_touched": num("gh_num_commits_on_files_touched"),
            "sloc": num("gh_sloc"),
            "test_lines_per_kloc": num("gh_test_lines_per_kloc"),
            "by_core_member": _bool(raw["gh_by_core_team_member"]),
        }
    )[keep.to_numpy()]
    df["n_files"] = df[["files_added", "files_deleted", "files_modified"]].sum(axis=1)
    df["test_code_ratio"] = df["test_churn"] / df["src_churn"].clip(lower=1)
    df = df.dropna(subset=["as_of"]).sort_values(["repo", "as_of"]).reset_index(drop=True)
    df["prev_build_failed"] = _prev_known_failure(df)
    return df


def _prev_known_failure(df: pd.DataFrame) -> pd.Series:
    """Most recent build in the same project whose outcome was known before this one started."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("repo", sort=False):
        done = g["completed_at"].to_numpy()
        labels = g["label"].astype(float).to_numpy()
        order = np.argsort(done, kind="stable")
        done_sorted, labels_sorted = done[order], labels[order]
        idx = np.searchsorted(done_sorted, g["as_of"].to_numpy(), side="left") - 1
        vals = np.where(idx >= 0, labels_sorted[np.clip(idx, 0, None)], np.nan)
        out.loc[g.index] = vals
    return out
