# Infrastructure runbook

```
GitHub ──webhook──► ALB ─/webhooks/*─► ingest (Fargate) ──► ElastiCache Redis (Streams + online store)
                     │                                        │
                     └──/v1/*───────► scorer (Fargate) ◄──────┤
                                          ▲                   ▼
                                          └─── worker (Fargate) ──► GitHub Check Runs
RDS Postgres ◄── all services      S3 (raw pages, log tails, datasets, MLflow artifacts)
MLflow (internal, service discovery)   EventBridge Scheduler: materialize nightly, monitor hourly
Secrets Manager → task env          GitHub Actions OIDC → deploy role (no stored AWS keys)
```

Cost-conscious defaults: one task per service, `db.t4g.micro`, `cache.t4g.micro`, no NAT
gateway (tasks get public IPs; their security group admits only the ALB). Destroy staging
when idle: `terraform destroy -var-file=env/staging.tfvars`.

## One-time bootstrap (per AWS account)

1. **State bucket** (versioned, private):
   ```bash
   aws s3api create-bucket --bucket preflight-tfstate-<account-id> --region ap-south-1 \
     --create-bucket-configuration LocationConstraint=ap-south-1
   aws s3api put-bucket-versioning --bucket preflight-tfstate-<account-id> \
     --versioning-configuration Status=Enabled
   ```
2. **First apply from your machine** (creates the GitHub OIDC provider + deploy role):
   ```bash
   cd infra/terraform
   cp env/staging.backend.hcl.example env/staging.backend.hcl   # fill the bucket name
   terraform init -backend-config=env/staging.backend.hcl
   terraform apply -var-file=env/staging.tfvars
   ```
   The first apply has no image yet: services start once CI pushes one (step 5).
3. **Secret values** (never in Terraform state), for each name in `terraform output secret_ids`:
   ```bash
   aws secretsmanager put-secret-value --secret-id preflight-staging/hash-salt --secret-string "$(openssl rand -base64 48)"
   aws secretsmanager put-secret-value --secret-id preflight-staging/github-app-private-key --secret-string "$(cat app.pem)"
   # github-webhook-secret, github-app-id, github-token, scorer-api-token, slack-webhook-url
   ```
4. **GitHub environments** (repo Settings → Environments): `staging` and `prod`
   (prod: add yourself as required reviewer).
   - secrets: `AWS_DEPLOY_ROLE_ARN` (= `terraform output deploy_role_arn`), `WEBHOOK_SECRET`,
     `SCORER_API_TOKEN`
   - variables: `AWS_REGION=ap-south-1`, `TF_STATE_BUCKET=preflight-tfstate-<account-id>`
5. **Enable deploys**: repo Settings → Variables → `DEPLOY_ENABLED=true`
   (until then, pushes to `main` run the checks and skip deploying).
6. **Push to `main`** → `deploy.yml`: checks → build → ECR → staging → migrate → smoke →
   approval → prod (same image) → smoke.
7. **Point the GitHub App webhook** at `terraform output webhook_url`.

## Day-2 operations

| Task | How |
|---|---|
| Roll back the model | Actions → ops → `rollback-model` (or `python scripts/rollback.py`) |
| Roll back the app | Actions → ops → `rollback-app` with a previous git SHA. Failed deploys also auto-roll back (ECS circuit breaker). |
| Retrain now | Actions → ops → `retrain` (normally triggered by the hourly monitor) |
| Backfill more history | Actions → ops → `backfill` |
| Promote a shadow | `python scripts/promote.py evaluate` then `promote <version>` |
| Watch | Grafana dashboard `Preflight` (local: `docker compose --profile ops up`), CloudWatch alarms → SNS email |
