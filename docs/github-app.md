# Registering the Preflight GitHub App

About 5 minutes of clicking. The App is only for **live webhooks and (from week 4) posting Check Runs**.
The historical backfill uses your personal token, because an App can only read repos it is installed on.

## 1. Get a public webhook URL for your laptop
GitHub cannot reach `localhost`, so relay through smee.io:

1. Open https://smee.io and click **Start a new channel**. Copy the URL (`https://smee.io/AbC123...`).
2. Run the relay (needs Node.js), leave it running:
   ```
   npx smee-client --url https://smee.io/AbC123 --target http://localhost:8000/webhooks/github
   ```

## 2. Create the App
GitHub → **Settings → Developer settings → GitHub Apps → New GitHub App**

| Field | Value |
|---|---|
| GitHub App name | `preflight-<your-handle>` (must be globally unique) |
| Homepage URL | your repo URL |
| Webhook → Active | ✅ |
| Webhook URL | the smee.io URL from step 1 |
| Webhook secret | the `GITHUB_WEBHOOK_SECRET` value from your `.env` |
| Where can this App be installed? | Only on this account |

**Repository permissions (least privilege, nothing else):**

| Permission | Access | Why |
|---|---|---|
| Actions | Read-only | workflow runs, jobs, logs, artifacts (labels) |
| Checks | Read & write | post the risk Check Run (week 4) |
| Contents | Read-only | commit diff stats and paths (never stored as code) |
| Metadata | Read-only | mandatory |
| Pull requests | Read-only | PR number, base SHA, changed files |

**Subscribe to events:** Pull request · Push · Workflow run · Workflow job · Check suite · Check run (for the Useful / Not useful buttons)

Click **Create GitHub App**.

## 3. Save the credentials
1. Note the **App ID** at the top of the App page.
2. **Private keys → Generate a private key**. Save the `.pem` to `C:\preflight\secrets\` (gitignored).
3. Add to `.env`:
   ```
   GITHUB_APP_ID=123456
   GITHUB_APP_PRIVATE_KEY_PATH=secrets/preflight.private-key.pem
   ```
   (Used from week 4 for short-lived installation tokens. In AWS it moves to Secrets Manager.)

## 4. Install and verify
1. App page → **Install App** → your account → pick a test repo (e.g. `kdt523/preflight` once pushed).
2. With `docker compose up` and smee running, open a PR on that repo.
3. You should see:
   - App page → **Advanced → Recent Deliveries**: green ✓ with response `202 {"status":"queued"}`
   - `docker compose logs ingest`: a JSON line with the delivery ID
   - `docker compose exec redis redis-cli XLEN gh:events` goes up
4. Click **Redeliver** on one delivery → response is `200 {"status":"duplicate"}` (dedup works).

## 5. Record real fixtures
Copy a few real payloads from **Recent Deliveries** into `tests/fixtures/webhooks/`
(`<event>.<action>.json`) to replace the hand-trimmed ones.
