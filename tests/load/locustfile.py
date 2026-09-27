"""Load test (SPEC §8): 50 webhooks/s against ingest, plus scorer traffic.

    uv run locust -f tests/load/locustfile.py --headless -u 60 -r 20 -t 2m \
        --host http://localhost:8000 --csv reports/load

WebhookUser paces itself to ~1 request/s, so 50 users ≈ 50 webhooks/s. ScorerUser hits
/v1/score on SCORER_HOST to measure p95 under load. Results land in reports/load_*.csv;
quote only numbers from those files.
"""

import json
import os
import random
import uuid
from pathlib import Path

from locust import HttpUser, between, constant_throughput, task

from apps.ingest.signature import sign

SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
FIXTURES = {
    p.stem.split(".")[0]: p.read_bytes()
    for p in (Path(__file__).parents[1] / "fixtures" / "webhooks").glob("*.json")
    if p.stem != "ping"
}


class WebhookUser(HttpUser):
    weight = 5
    wait_time = constant_throughput(1)  # ~1 webhook/s per user

    @task
    def webhook(self) -> None:
        event = random.choice(list(FIXTURES))
        body = FIXTURES[event]
        self.client.post(
            "/webhooks/github",
            data=body,
            name=f"webhook:{event}",
            headers={
                "X-GitHub-Event": event,
                "X-GitHub-Delivery": str(uuid.uuid4()),
                "X-Hub-Signature-256": sign(SECRET, body),
                "Content-Type": "application/json",
            },
        )


class ScorerUser(HttpUser):
    weight = 1
    host = os.environ.get("SCORER_HOST", "http://localhost:8001")
    wait_time = between(0.2, 0.5)

    @task
    def score(self) -> None:
        n = random.randint(1, 30)
        body = {
            "repo": os.environ.get("LOAD_REPO", "acme/app"),
            "sha": uuid.uuid4().hex + "0" * 8,
            "files": [
                {
                    "path": f"src/mod{random.randint(0, 50)}.py",
                    "added": random.randint(1, 200),
                    "deleted": random.randint(0, 80),
                }
                for _ in range(n)
            ],
            "log_prediction": False,
        }
        token = os.environ.get("SCORER_API_TOKEN")
        self.client.post(
            "/v1/score",
            data=json.dumps(body),
            name="score",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
