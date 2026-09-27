"""Features computed from the change itself (size, spread, change type). No history needed."""

import math
import posixpath
import re
from dataclasses import dataclass

LOCKFILE = re.compile(
    r"(^|/)(poetry\.lock|uv\.lock|Pipfile\.lock|package-lock\.json|yarn\.lock|pnpm-lock\.yaml"
    r"|Cargo\.lock|go\.sum|Gemfile\.lock|composer\.lock|requirements[^/]*\.lock|pdm\.lock)$"
)
CI_CONFIG = re.compile(
    r"^\.github/workflows/|(^|/)(\.gitlab-ci\.yml|\.travis\.yml|azure-pipelines\.yml|tox\.ini"
    r"|noxfile\.py|\.pre-commit-config\.yaml|Jenkinsfile)$|^\.circleci/"
)
MIGRATION = re.compile(r"(^|/)(migrations|alembic|migrate)/")
DEP_MANIFEST = re.compile(
    r"(^|/)(pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.(txt|in)|environment\.ya?ml"
    r"|package\.json|Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle(\.kts)?)$"
)
DOCS = re.compile(r"(^|/)docs?/|\.(md|rst|txt|adoc)$|(^|/)(LICENSE|AUTHORS|CHANGES|CHANGELOG)")


@dataclass(frozen=True)
class FileChange:
    path: str
    added: int
    deleted: int
    change_type: str = "modified"  # added|modified|removed|renamed|...
    is_test: bool = False


def change_entropy(churns: list[int]) -> float:
    """Hassan (ICSE 2009) entropy of changes, normalised to [0, 1] by log2(n)."""
    total = sum(churns)
    n = sum(1 for c in churns if c > 0)
    if total <= 0 or n <= 1:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in churns if c > 0)
    return h / math.log2(n)


def diff_features(files: list[FileChange], truncated: bool = False) -> dict[str, float]:
    added = sum(f.added for f in files)
    deleted = sum(f.deleted for f in files)
    n = len(files)
    dirs = {posixpath.dirname(f.path) for f in files}
    return {
        "n_files": n,
        "lines_added": added,
        "lines_deleted": deleted,
        "log_churn": math.log1p(added + deleted),
        "n_dirs": len(dirs),
        "change_entropy": change_entropy([f.added + f.deleted for f in files]),
        "files_truncated": float(truncated),
        "touches_lockfile": float(any(LOCKFILE.search(f.path) for f in files)),
        "touches_ci_config": float(any(CI_CONFIG.search(f.path) for f in files)),
        "touches_migrations": float(any(MIGRATION.search(f.path) for f in files)),
        "touches_dep_manifest": float(any(DEP_MANIFEST.search(f.path) for f in files)),
        "n_new_files": sum(f.change_type == "added" for f in files),
        "n_removed_files": sum(f.change_type == "removed" for f in files),
        "n_renamed_files": sum(f.change_type == "renamed" for f in files),
        "docs_only": float(n > 0 and all(DOCS.search(f.path) for f in files)),
        "frac_test_files": (sum(f.is_test for f in files) / n) if n else 0.0,
    }


def test_basenames_for(path: str) -> set[str]:
    """Conventional test file names for a source file: foo.py -> test_foo.py, foo_test.py."""
    base = posixpath.basename(path)
    stem, ext = posixpath.splitext(base)
    if not ext:
        return set()
    return {f"test_{stem}{ext}", f"{stem}_test{ext}", f"{stem}.test{ext}", f"{stem}.spec{ext}"}
