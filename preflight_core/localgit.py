"""Local git helpers for the CLI and pytest plugin. Reads diff *stats* only, never content."""

import re
import subprocess
from pathlib import Path

_REMOTE = re.compile(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$")


def _git(*args: str, cwd: str | Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()


def repo_slug(cwd: str | Path | None = None) -> str | None:
    try:
        m = _REMOTE.search(_git("remote", "get-url", "origin", cwd=cwd))
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return m.group(1) if m else None


def merge_base(base: str = "origin/main", cwd: str | Path | None = None) -> str | None:
    try:
        return _git("merge-base", "HEAD", base, cwd=cwd)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def parse_numstat(out: str) -> list[dict]:
    files: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        if " => " in path:  # rename: "dir/{old => new}.py" or "old => new"
            path = re.sub(r"\{[^{}]*? => ([^{}]*?)\}", r"\1", path)
            path = path.split(" => ")[-1]
        f = files.setdefault(
            path, {"path": path, "added": 0, "deleted": 0, "change_type": "modified"}
        )
        # binary files report "-"
        f["added"] += int(added) if added.isdigit() else 0
        f["deleted"] += int(deleted) if deleted.isdigit() else 0
    return list(files.values())


def local_diff(base: str = "origin/main", cwd: str | Path | None = None) -> list[dict]:
    """Committed changes since the merge-base with `base`, plus uncommitted edits."""
    mb = merge_base(base, cwd)
    commands = [("diff", "--numstat", "-M", "HEAD")]
    if mb:
        commands.insert(0, ("diff", "--numstat", "-M", mb, "HEAD"))
    out = []
    for cmd in commands:  # not a repo / no commits / shallow clone: use what is available
        try:
            out.append(_git(*cmd, cwd=cwd))
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    merged: dict[str, dict] = {}
    for f in parse_numstat("\n".join(o for o in out if o)):
        m = merged.setdefault(f["path"], {**f, "added": 0, "deleted": 0})
        m["added"] += f["added"]
        m["deleted"] += f["deleted"]
    return list(merged.values())


def head_sha(cwd: str | Path | None = None) -> str:
    return _git("rev-parse", "HEAD", cwd=cwd)


def author_login(cwd: str | Path | None = None) -> str | None:
    try:
        return _git("config", "user.email", cwd=cwd) or None
    except subprocess.SubprocessError:
        return None
