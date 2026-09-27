import io
import zipfile

import pytest
from defusedxml import EntitiesForbidden

from preflight_core.backfill import junit
from preflight_core.backfill.junit import junit_from_zip, parse_junit

XML = b"""<?xml version="1.0"?>
<testsuites><testsuite name="s">
  <testcase classname="tests.test_a" name="test_ok" time="0.012"/>
  <testcase classname="tests.test_a" name="test_bad" time="1.5"><failure message="x"/></testcase>
  <testcase classname="tests.test_a" name="test_err"><error/></testcase>
  <testcase classname="tests.test_a" name="test_skip"><skipped/></testcase>
</testsuite></testsuites>"""


def zipped(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_parse_statuses_and_durations():
    rows = {r["test_id"]: r for r in parse_junit(XML)}
    assert rows["tests.test_a::test_ok"] == {
        "test_id": "tests.test_a::test_ok",
        "status": "passed",
        "duration_ms": 12,
    }
    assert rows["tests.test_a::test_bad"]["status"] == "failed"
    assert rows["tests.test_a::test_err"]["status"] == "error"
    assert rows["tests.test_a::test_skip"]["status"] == "skipped"


def test_zip_merges_shards_keeping_worst_status():
    shard2 = XML.replace(b'name="test_ok" time="0.012"/>', b'name="test_ok"><failure/></testcase>')
    rows = {
        r["test_id"]: r
        for r in junit_from_zip(
            zipped(
                {"py311.xml": XML, "py312.xml": shard2, "coverage.txt": b"x", "bad.xml": b"<nope"}
            )
        )
    }
    assert rows["tests.test_a::test_ok"]["status"] == "failed"
    assert len(rows) == 4


def test_rejects_entity_expansion():
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><testsuite>&a;</testsuite>'
    with pytest.raises(EntitiesForbidden):
        parse_junit(bomb)


def test_rejects_zip_bomb(monkeypatch):
    monkeypatch.setattr(junit, "MAX_UNZIPPED_BYTES", 10)
    with pytest.raises(ValueError):
        junit_from_zip(zipped({"a.xml": XML}))
