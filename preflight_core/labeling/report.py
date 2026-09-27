"""Label breakdown for the week-1 exit check. Numbers come straight from the database."""

from sqlalchemy import Engine, text

TYPES = ["pass", "code_defect", "flaky", "infra", "config", "cancelled"]

_BY_REPO = text(
    """
    SELECT r.full_name, b.failure_type, count(*) AS n
    FROM builds b JOIN repos r ON r.id = b.repo_id
    WHERE b.failure_type IS NOT NULL
    GROUP BY 1, 2
    """
)
_SOURCES = text(
    """
    SELECT failure_type, label_source, count(*) FROM builds
    WHERE failure_type NOT IN ('pass', 'cancelled') GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """
)
_SHAS = text(
    """
    SELECT count(*) FILTER (WHERE has_code) AS failing, count(*) AS total FROM (
      SELECT repo_id, sha, bool_or(failure_type = 'code_defect') AS has_code
      FROM builds WHERE failure_type IN ('pass', 'code_defect') GROUP BY 1, 2
    ) s
    """
)
_COVERAGE = text(
    """
    SELECT count(*) FILTER (
             WHERE conclusion IN ('failure', 'timed_out', 'startup_failure')) AS failed,
           count(*) FILTER (WHERE log_available) AS with_logs,
           (SELECT count(DISTINCT build_id) FROM test_results) AS with_junit,
           (SELECT count(*) FROM builds b WHERE NOT EXISTS (
               SELECT 1 FROM commits c WHERE c.repo_id = b.repo_id AND c.sha = b.sha)) AS no_diff
    FROM builds
    """
)


def build_report(engine: Engine) -> str:
    with engine.connect() as c:
        by_repo: dict[str, dict[str, int]] = {}
        for repo, ft, n in c.execute(_BY_REPO):
            by_repo.setdefault(repo, {})[ft] = n
        sources = c.execute(_SOURCES).all()
        failing_shas, total_shas = c.execute(_SHAS).one()
        cov = c.execute(_COVERAGE).one()

    lines = [f"{'repo':40} " + " ".join(f"{t[:11]:>11}" for t in TYPES) + f" {'rate':>6}"]
    totals = dict.fromkeys(TYPES, 0)
    for repo in sorted(by_repo):
        d = by_repo[repo]
        for t in TYPES:
            totals[t] += d.get(t, 0)
        lines.append(
            f"{repo:40} " + " ".join(f"{d.get(t, 0):>11}" for t in TYPES) + f" {_rate(d):>6}"
        )
    lines.append(
        f"{'TOTAL':40} " + " ".join(f"{totals[t]:>11}" for t in TYPES) + f" {_rate(totals):>6}"
    )

    all_builds = sum(totals.values())
    usable = totals["pass"] + totals["code_defect"]
    excluded = totals["flaky"] + totals["infra"]
    failures = sum(totals[t] for t in ("code_defect", "flaky", "infra", "config"))
    lines += [
        "",
        f"builds (run attempts) labelled : {all_builds}",
        f"training rows (pass+code_defect): {usable}",
        f"noisy failures removed         : {excluded} of {failures} failures "
        f"({excluded / failures:.1%})"
        if failures
        else "noisy failures removed: n/a",
        f"distinct SHAs (pass/code)      : {total_shas}  failing: {failing_shas}",
        f"failed builds with log tails   : {cov.with_logs} of {cov.failed}",
        f"builds with JUnit results      : {cov.with_junit}",
        f"builds without a diff          : {cov.no_diff}",
        "",
        "label sources (non-pass):",
    ]
    lines += [f"  {ft:12} {src:10} {n}" for ft, src, n in sources]
    return "\n".join(lines)


def _rate(d: dict[str, int]) -> str:
    denom = d.get("pass", 0) + d.get("code_defect", 0)
    return f"{d.get('code_defect', 0) / denom:.1%}" if denom else "-"
