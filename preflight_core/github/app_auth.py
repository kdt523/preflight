"""GitHub App authentication: App JWT -> short-lived installation tokens (cached).

Installation tokens live 1 hour; we refresh 5 minutes early. The private key comes from a
file locally and from Secrets Manager (as an env var) in AWS.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import jwt

from preflight_core.github.client import API_VERSION, GitHubClient

REFRESH_MARGIN_S = 300


def app_jwt(app_id: str, private_key: str, now: float | None = None) -> str:
    now = int(now or time.time())
    # iat backdated 60 s for clock drift; GitHub caps exp at 10 minutes.
    payload = {"iat": now - 60, "exp": now + 540, "iss": str(app_id)}
    return jwt.encode(payload, private_key, algorithm="RS256")


def load_private_key(path: str | None, inline: str | None) -> str:
    if inline:
        return inline.replace("\\n", "\n")
    if path and Path(path).exists():
        return Path(path).read_text()
    raise RuntimeError("GitHub App private key not configured")


class InstallationTokens:
    def __init__(
        self,
        app_id: str,
        private_key: str,
        api_url: str = "https://api.github.com",
        transport: httpx.BaseTransport | None = None,
        clock=time.time,
    ) -> None:
        self.app_id = app_id
        self.private_key = private_key
        self.api_url = api_url
        self._transport = transport
        self._clock = clock
        self._cache: dict[int, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def token(self, installation_id: int) -> str:
        with self._lock:
            hit = self._cache.get(installation_id)
            if hit and hit[1] - self._clock() > REFRESH_MARGIN_S:
                return hit[0]
            with httpx.Client(base_url=self.api_url, transport=self._transport, timeout=10) as c:
                r = c.post(
                    f"/app/installations/{installation_id}/access_tokens",
                    headers={
                        "Authorization": f"Bearer {app_jwt(self.app_id, self.private_key)}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": API_VERSION,
                    },
                )
                r.raise_for_status()
                data = r.json()
            expires = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
            self._cache[installation_id] = (data["token"], expires)
            return data["token"]

    def client(self, installation_id: int, **kw) -> GitHubClient:
        return GitHubClient(self.token(installation_id), self.api_url, **kw)
