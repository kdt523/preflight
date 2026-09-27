"""Sign and POST webhook fixtures to a running ingest service.

uv run python scripts/replay_webhooks.py                   # all fixtures, fresh delivery IDs
uv run python scripts/replay_webhooks.py --same-id         # stable IDs: 2nd run tests dedup
uv run python scripts/replay_webhooks.py --url https://staging.example/webhooks/github
"""

import argparse
import uuid
from pathlib import Path

import httpx

from apps.ingest.signature import sign
from preflight_core.config import get_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/webhooks/github")
    ap.add_argument("--dir", default="tests/fixtures/webhooks")
    ap.add_argument("--same-id", action="store_true", help="derive delivery id from file name")
    args = ap.parse_args()

    secret = get_settings().github_webhook_secret.get_secret_value()
    for path in sorted(Path(args.dir).glob("*.json")):
        body = path.read_bytes()
        delivery = (
            str(uuid.uuid5(uuid.NAMESPACE_URL, path.name)) if args.same_id else str(uuid.uuid4())
        )
        r = httpx.post(
            args.url,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": path.stem.split(".")[0],
                "X-GitHub-Delivery": delivery,
                "X-Hub-Signature-256": sign(secret, body),
            },
        )
        print(f"{path.name:32} {r.status_code} {r.text}")


if __name__ == "__main__":
    main()
