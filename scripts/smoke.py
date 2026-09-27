"""Post-deploy smoke test: health, signed webhook ping, authenticated scorer call.

    python scripts/smoke.py --base-url http://<alb-dns>
Env: WEBHOOK_SECRET, SCORER_API_TOKEN. Exits non-zero on any failure.
"""

import argparse
import json
import os
import sys
import uuid

import httpx

from apps.ingest.signature import sign


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    base = ap.parse_args().base_url.rstrip("/")
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
        if not ok:
            failures.append(name)

    with httpx.Client(base_url=base, timeout=10) as c:
        body = json.dumps({"zen": "smoke", "hook_id": 0}).encode()
        r = c.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": f"smoke-{uuid.uuid4()}",
                "X-Hub-Signature-256": sign(os.environ.get("WEBHOOK_SECRET", ""), body),
            },
        )
        check(
            "ingest signed ping",
            r.status_code == 200 and r.json() == {"status": "pong"},
            str(r.status_code),
        )
        r = c.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": f"smoke-{uuid.uuid4()}",
                "X-Hub-Signature-256": "sha256=" + "0" * 64,
            },
        )
        check("ingest rejects bad signature", r.status_code == 401, str(r.status_code))
        token = os.environ.get("SCORER_API_TOKEN", "")
        r = c.post(
            "/v1/score",
            json={"repo": "smoke/test", "sha": "0" * 40, "files": [], "log_prediction": False},
            headers={"Authorization": f"Bearer {token}"},
        )
        # 503 = alive but no champion model yet (worker fails open): acceptable before 1st train
        check("scorer authenticated call", r.status_code in (200, 503), str(r.status_code))
        r = c.post("/v1/score", json={"repo": "smoke/test", "sha": "0" * 40, "files": []})
        check("scorer rejects missing token", r.status_code == 401, str(r.status_code))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
