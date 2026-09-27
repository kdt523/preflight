import pandas as pd

from preflight_core.data.travistorrent import TT_FEATURES, to_frame


def raw(rows):
    base = {
        "gh_project_name": "a/b",
        "gh_is_pr": "false",
        "tr_log_bool_tests_failed": "false",
        "git_diff_src_churn": 10,
        "git_diff_test_churn": 5,
        "gh_diff_files_added": 1,
        "gh_diff_files_deleted": 0,
        "gh_diff_files_modified": 2,
        "gh_diff_tests_added": 0,
        "gh_diff_src_files": 2,
        "gh_diff_doc_files": 0,
        "gh_diff_other_files": 1,
        "gh_num_commits_in_push": 1,
        "gh_team_size": 4,
        "gh_num_commits_on_files_touched": 3,
        "gh_sloc": 1000,
        "gh_test_lines_per_kloc": 200,
        "gh_by_core_team_member": "true",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_status_mapping_and_prev_failure_is_point_in_time():
    df = to_frame(
        raw(
            [
                # build 1 fails, finishes at 10:30
                {
                    "tr_build_id": 1,
                    "tr_status": "failed",
                    "tr_log_bool_tests_failed": "true",
                    "gh_build_started_at": "2016-01-01T10:00:00Z",
                    "tr_duration": 1800,
                    "git_trigger_commit": "a",
                },
                # build 2 starts 10:10: build 1's outcome not known yet
                {
                    "tr_build_id": 2,
                    "tr_status": "passed",
                    "gh_build_started_at": "2016-01-01T10:10:00Z",
                    "tr_duration": 60,
                    "git_trigger_commit": "b",
                },
                # build 3 at 11:00: latest completion is build 1 (10:30, failed)
                {
                    "tr_build_id": 3,
                    "tr_status": "errored",
                    "gh_build_started_at": "2016-01-01T11:00:00Z",
                    "tr_duration": 60,
                    "git_trigger_commit": "c",
                },
                {
                    "tr_build_id": 4,
                    "tr_status": "canceled",
                    "gh_build_started_at": "2016-01-01T12:00:00Z",
                    "tr_duration": 60,
                    "git_trigger_commit": "d",
                },
            ]
        )
    )
    assert list(df["build_id"]) == [1, 2, 3]  # canceled dropped
    assert list(df["label"].astype("float").fillna(-1)) == [1, 0, -1]  # errored excluded
    prev = df.set_index("build_id")["prev_build_failed"]
    assert pd.isna(prev[1]) and pd.isna(prev[2])  # nothing known yet at 10:10
    assert prev[3] == 1.0  # build 1 completed last (10:30) before 11:00
    assert set(TT_FEATURES) <= set(df.columns)
