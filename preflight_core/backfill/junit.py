"""JUnit XML from CI artifacts -> test_results rows. Artifacts are untrusted input."""

import io
import zipfile
from typing import Any

from defusedxml import ElementTree

MAX_UNZIPPED_BYTES = 50 * 1024 * 1024


def parse_junit(xml: bytes) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml)
    out = []
    for case in root.iter("testcase"):
        cls, name = case.get("classname") or "", case.get("name") or ""
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        try:
            duration_ms = round(float(case.get("time") or 0) * 1000)
        except ValueError:
            duration_ms = None
        out.append(
            {
                "test_id": f"{cls}::{name}" if cls else name,
                "status": status,
                "duration_ms": duration_ms,
            }
        )
    return out


def junit_from_zip(blob: bytes) -> list[dict[str, Any]]:
    """Parse every JUnit XML in an artifact zip. Refuses zip bombs; skips non-JUnit XML."""
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED_BYTES:
            raise ValueError("artifact too large when unzipped")
        for info in zf.infolist():
            if not info.filename.lower().endswith(".xml"):
                continue
            try:
                rows.extend(parse_junit(zf.read(info)))
            except ElementTree.ParseError:
                continue
    return merge_results(rows)


def merge_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per test id; a test repeated across files or matrix shards keeps its worst status."""
    results: dict[str, dict[str, Any]] = {}
    for r in rows:
        prev = results.get(r["test_id"])
        if prev is None or _rank(r["status"]) > _rank(prev["status"]):
            results[r["test_id"]] = r
    return list(results.values())


def _rank(status: str) -> int:
    return {"skipped": 0, "passed": 1, "error": 2, "failed": 3}[status]
