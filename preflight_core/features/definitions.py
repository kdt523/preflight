"""Single source of truth for features: names, family, monotone direction, actionability.

Training, serving, explanations, what-ifs and ablations all read this list, so a feature
cannot silently differ between them. Bump FEATURE_SET_VERSION when semantics change.
"""

from dataclasses import dataclass

FEATURE_SET_VERSION = "v1"

SIZE, CHANGE_TYPE, FILE_HISTORY, COCHANGE, TEST_SIGNAL, CONTEXT, AUTHOR = (
    "size",
    "change_type",
    "file_history",
    "cochange",
    "test_signal",
    "context",
    "author",
)
FAMILIES = (SIZE, CHANGE_TYPE, FILE_HISTORY, COCHANGE, TEST_SIGNAL, CONTEXT, AUTHOR)
HIDDEN_FAMILIES = {AUTHOR}  # used by the model, never shown in UI/API explanations


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    monotone: int = 0  # +1: more can only raise risk, -1: only lower, 0: unconstrained
    actionable: bool = False
    doc: str = ""
    # PSI-monitored? Off for cumulative counters and per-repo scalars, which move by
    # construction (history grows, repo base rate converges); live PR-AUC covers them.
    drift: bool = True


FEATURES: tuple[Feature, ...] = (
    # size & spread
    Feature("n_files", SIZE, +1, True, "files changed"),
    Feature("lines_added", SIZE, +1, True, "lines added"),
    Feature("lines_deleted", SIZE, +1, True, "lines deleted"),
    Feature("log_churn", SIZE, +1, True, "log(1 + lines added + deleted)"),
    Feature("n_dirs", SIZE, +1, True, "distinct directories touched"),
    Feature("change_entropy", SIZE, +1, True, "normalised entropy of churn across files"),
    Feature("files_truncated", SIZE, 0, False, "diff listing was truncated by the API"),
    # change type
    Feature("touches_lockfile", CHANGE_TYPE, 0, False, "lockfile changed"),
    Feature("touches_ci_config", CHANGE_TYPE, 0, False, "CI config changed"),
    Feature("touches_migrations", CHANGE_TYPE, 0, False, "DB migration changed"),
    Feature("touches_dep_manifest", CHANGE_TYPE, 0, False, "dependency manifest changed"),
    Feature("n_new_files", CHANGE_TYPE, 0, False, "files added"),
    Feature("n_removed_files", CHANGE_TYPE, 0, False, "files removed"),
    Feature("n_renamed_files", CHANGE_TYPE, 0, False, "files renamed"),
    Feature("docs_only", CHANGE_TYPE, 0, False, "only docs changed"),
    Feature("frac_test_files", CHANGE_TYPE, 0, False, "share of changed files that are tests"),
    # file history (point-in-time, Bayesian-smoothed)
    Feature("max_file_fail_rate", FILE_HISTORY, +1, False, "riskiest touched file's fail rate"),
    Feature("mean_file_fail_rate", FILE_HISTORY, +1, False, "mean touched-file fail rate"),
    Feature("sum_decayed_churn", FILE_HISTORY, 0, False, "30-day decayed churn of touched files"),
    Feature("hotspot_score", FILE_HISTORY, +1, False, "max decayed churn x fail rate"),
    Feature("frac_files_unseen", FILE_HISTORY, 0, False, "share of files with no history"),
    Feature("file_rate_lift", FILE_HISTORY, +1, False, "riskiest file rate / repo base rate"),
    # co-change
    Feature("missing_partners", COCHANGE, +1, True, "usual co-changed files absent here"),
    Feature("max_missing_confidence", COCHANGE, +1, True, "strongest missing co-change rule"),
    # test signal
    Feature("test_code_ratio", TEST_SIGNAL, -1, True, "test lines added / code lines added"),
    Feature("touches_tests", TEST_SIGNAL, -1, True, "any test file changed"),
    Feature("frac_src_with_tests", TEST_SIGNAL, -1, True, "changed source files with a known test"),
    # context
    Feature("is_pr", CONTEXT, 0, False, "pull request build"),
    Feature("is_default_branch", CONTEXT, 0, False, "push to the default branch"),
    Feature("pushes_since_base", CONTEXT, +1, True, "default-branch pushes since PR base"),
    Feature("main_red", CONTEXT, +1, False, "default branch currently failing"),
    Feature("hours_since_last_green", CONTEXT, 0, False, "hours since default branch passed"),
    Feature("concurrent_same_files", CONTEXT, +1, False, "in-flight builds touching same files"),
    Feature(
        "repo_recent_fail_rate",
        CONTEXT,
        +1,
        False,
        "smoothed fail rate of last 50 builds",
        drift=False,
    ),
    Feature(
        "repo_base_rate",
        CONTEXT,
        +1,
        False,
        "repo's smoothed failure base rate so far",
        drift=False,
    ),
    # author (model only)
    Feature("author_prior_builds", AUTHOR, 0, False, "author's prior builds", drift=False),
    Feature("author_file_familiarity", AUTHOR, 0, False, "share of files author touched before"),
    Feature("author_fail_rate", AUTHOR, 0, False, "author's smoothed fail rate"),
)

BY_NAME = {f.name: f for f in FEATURES}


def feature_names(exclude_families: tuple[str, ...] | set[str] = ()) -> list[str]:
    return [f.name for f in FEATURES if f.family not in exclude_families]


def drift_features() -> list[str]:
    return [f.name for f in FEATURES if f.drift]


def monotone_constraints(names: list[str]) -> list[int]:
    return [BY_NAME[n].monotone for n in names]


def family_of(name: str) -> str:
    return BY_NAME[name].family
