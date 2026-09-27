from preflight_core.backfill.repo_selection import RepoStats, eligible, select


def stats(name, runs, failures, junit=False, archived=False):
    return RepoStats(name, "main", archived, runs, failures, junit)


def test_ineligible_repos_are_filtered():
    assert not eligible(stats("a/tiny", 100, 10))  # too few runs
    assert not eligible(stats("a/red", 1000, 800))  # permanently red, labels are noise
    assert not eligible(stats("a/green", 1000, 5))  # almost no positives
    assert not eligible(stats("a/old", 5000, 500, archived=True))
    assert eligible(stats("a/ok", 1000, 100))


def test_select_prefers_volume_failures_and_junit():
    pool = [
        stats("a/small", 600, 60),
        stats("a/big", 20000, 2000),
        stats("a/junit", 600, 60, junit=True),
    ]
    assert [s.full_name for s in select(pool, n=2)] == ["a/big", "a/junit"]
