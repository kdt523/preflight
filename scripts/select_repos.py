"""Measure candidate repos and write the top N to repos.yaml.

uv run python scripts/select_repos.py [--n 20]
"""

import argparse
from pathlib import Path

import yaml

from preflight_core.backfill.repo_selection import measure_all, select
from preflight_core.config import get_settings
from preflight_core.github.client import GitHubClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--candidates", default="repos.candidates.yaml")
    ap.add_argument("--out", default="repos.yaml")
    args = ap.parse_args()

    s = get_settings()
    if not s.github_token.get_secret_value():
        raise SystemExit("GITHUB_TOKEN is empty in .env")
    names = yaml.safe_load(Path(args.candidates).read_text())["candidates"]
    gh = GitHubClient(s.github_token.get_secret_value(), s.github_api_url)
    stats, errors = measure_all(gh, names)
    chosen = select(stats, args.n)

    print(f"{'repo':42} {'runs/90d':>9} {'fail':>7} {'rate':>6} junit")
    for st in sorted(stats, key=lambda x: x.runs, reverse=True):
        mark = "*" if st in chosen else " "
        print(
            f"{mark}{st.full_name:41} {st.runs:9d} {st.failures:7d} "
            f"{st.failure_rate:6.1%} {'yes' if st.junit_artifacts else '-'}"
        )
    for name, err in errors.items():
        print(f"!{name}: {err}")
    Path(args.out).write_text(
        yaml.safe_dump({"repos": [c.as_dict() for c in chosen]}, sort_keys=False)
    )
    print(f"\n{len(chosen)} repos -> {args.out}  ({gh.calls} API calls)")


if __name__ == "__main__":
    main()
