# Preflight — Specification

## 1. Product features
1. **Risk check on every PR** — GitHub Check Run with calibrated P(fail) and band (low < 0.25 ≤ watch < 0.55 ≤ high; thresholds tuned per repo). Posted before first CI job finishes.
2. **Reasons in plain language** — top SHAP contributions → sentence templates ("auth/session.py broke 38% of builds that touched it in 90 days").
3. **What-if fixes** — re-score with actionable features edited (tests added, rebased, PR split); show risk delta.
4. **Fail-fast test order** — per-test model ranks tests; pytest plugin reorders suite. Metrics: APFD, time to first failure.
5. **Adaptive CI plan** — low risk: lint + affected tests; high risk: full matrix; pushes to main always full.
6. **Flaky test registry** — flakiness score per test, quarantine suggestion; flaky runs never become labels.
7. **Pre-push CLI** — `preflight check` scores local diff via the same API.
8. **Dashboard** — hotspot treemap (churn × failure rate), CI minutes saved, calibration curve, drift status.
9. **Feedback** — useful / not useful votes on checks; weekly Slack digest.

## 2. Architecture (three loops)
- **Predict (latency path):** GitHub webhook → Ingest API (HMAC verify, dedup by X-GitHub-Delivery) → Redis Streams → Feature builder (point-in-time, Redis online store) → Risk scorer (LightGBM + SHAP + conformal) → Check Run.
- **Learn (async):** workflow_run completed → Labeler (taxonomy) → Feature store (S3 Parquet + Postgres) → Trainer (temporal CV, calibration, threshold) → MLflow registry (shadow → champion) → scorer.
- **Watch:** dev feedback → labeler; Monitor (PSI drift, rolling PR-AUC) → retrain trigger; dashboard + Slack digest.
- Nightly batch: history aggregates from feature store → online store.

## 3. Data sources
- GitHub Actions API on 20–30 active OSS repos: runs, jobs, reruns, logs, commit diffs (stats/paths only)
- Test artifacts: JUnit XML, coverage.xml where available
- TravisTorrent (public research dataset) for global-model pretraining / benchmark
- Preflight's own CI history (dogfooding)

## 4. Failure taxonomy
| Class | Detection | Treatment |
|---|---|---|
| code_defect | assertion/compile error, fails again on rerun of same SHA | positive label |
| flaky | same SHA fails then passes on rerun | excluded → flaky registry |
| infra | log regex: runner lost, OOM, timeout, network, rate limit | excluded |
| config | CI YAML / dependency resolution errors | own class (v2) |
| cancelled | superseded / manual cancel | dropped |

Log classifier: regex rules first; TF-IDF + logistic regression on last 200 log lines for the rest. Hand-label ~300 logs to measure precision.

## 5. Feature families (all as of build start)
| Family | Examples |
|---|---|
| Size & spread | files, lines ±, dirs touched, change entropy |
| Change type | lockfile, CI YAML, migrations, public signature changes |
| File history | smoothed per-file failure rate, 30-day decayed churn, hotspot score |
| Co-change | missing partners (files co-changed ≥ 60% of the time, absent here) |
| Test signal | changed-line coverage, test lines ÷ code lines added, mapped tests exist |
| Context | commits behind main, main currently red, concurrent PRs on same files, hours since last green |
| Author (model only) | prior commits to these files, smoothed recent failure rate — never displayed |

## 6. Data model (Postgres)
- repos(id, installation_id, default_branch, failure_base_rate)
- commits(sha, repo_id, author_hash, committed_at, parent_sha)
- file_changes(sha, path, added, deleted, change_type, is_test)
- builds(id, repo_id, sha, pr_number, started_at, conclusion, failure_type, rerun_of)
- test_results(build_id, test_id, status, duration_ms, flaky_score)
- features(repo_id, sha, as_of, feature_set_version, vector jsonb)
- predictions(id, repo_id, sha, model_version, p_fail, band, conformal_set, shap_top jsonb, latency_ms)
- feedback(prediction_id, user_hash, useful)
- model_runs(version, data_window, pr_auc, ece, threshold, status)
- Row-level security by installation_id.

## 7. Repo layout
```
preflight/
├── apps/{ingest,worker,scorer,dashboard}/
├── preflight_core/{features,labeling,models,monitoring}/
├── cli/
├── pytest_plugin/
├── infra/terraform/
├── .github/workflows/
└── tests/  (unit, integration, leakage, model behaviour)
```

## 8. Production checklist
- Reliability: fail-open (>2 s or error → full CI, check "skipped"); idempotent webhooks; retries + DLQ; ETag conditional requests + token bucket; resumable backfill
- MLOps: shared feature defs + train/serve parity test; MLflow tags (data window, feature hash, metrics); 7-day shadow mode; PSI drift; weekly + triggered retrain; one-command rollback; Pandera validation
- Security: least-privilege GitHub App; HMAC-SHA256; no source stored; Secrets Manager; short-lived tokens; RLS; blame-free mode
- Observability: OpenTelemetry traces; Prometheus metrics (latency, queue lag, fail-open count, prediction distribution); Grafana alerts; JSON logs by delivery ID
- Testing: recorded webhook fixtures; leakage test; behavioural tests (adding tests never raises risk; whitespace-only diff stays low); Locust load test at 50 webhooks/s
- Delivery: Terraform (ECS Fargate, RDS, ElastiCache, S3, Secrets Manager); GitHub Actions lint → test → build → ECR → staging → smoke → prod; Preflight scores its own PRs; Docker Compose + webhook replayer

## 9. Build plan
### Week 1 — Data foundation
GitHub App registration, webhook receiver, backfill (runs, jobs, diffs, JUnit) from 20 repos into Postgres + S3, labeler v0.
**Exit:** 50k+ labelled builds with failure-type breakdown.
### Week 2 — Features + baseline
Point-in-time pipeline for all 7 families, Bayesian smoothing, co-change mining, leakage test, logistic regression under temporal CV.
**Exit:** baseline PR-AUC vs base rate.
### Week 3 — Model + explanations
LightGBM (monotonic), isotonic calibration, cost threshold, conformal sets, SHAP templates, what-ifs, per-test model.
**Exit:** eval notebook with ablations + reliability curve.
### Week 4 — Serving path
Webhook → queue → features → FastAPI scorer → Check Run; Redis online store; fail-open; pytest plugin.
**Exit:** live check on a real PR in < 2 s.
### — MVP cut line —
### Week 5 — Operate
MLflow registry, shadow deploys, drift monitor, retraining, Terraform on AWS, GitHub Actions CI/CD, Prometheus + Grafana.
**Exit:** one-command deploy/rollback; drift alert fires in a test.
### Week 6 — Polish
Dashboard, CLI, Slack digest, load test, README with architecture + results table, 2-min demo video.
**Exit:** public repo understandable in 60 s.

## 10. Resume bullets (fill [ ] with measured numbers only)
**Preflight — ML-Based CI Build Failure Forecasting** | Python, LightGBM, SHAP, FastAPI, Redis, PostgreSQL, MLflow, AWS (ECS, RDS, S3), Terraform, GitHub Actions
- Built a GitHub App that predicts CI build failure before the pipeline runs, trained on [N] builds from [M] open-source repositories; calibrated LightGBM reached PR-AUC [X] vs [Y] for a logistic-regression baseline under rolling temporal validation.
- Designed a failure taxonomy (code defect, flaky, infra, config) with a log classifier that removed [X]% noisy labels, plus a point-in-time feature pipeline (change entropy, Bayesian-smoothed file failure rates, missing co-change, diff coverage) guarded by automated leakage tests.
- Shipped SHAP explanations and counterfactual what-if fixes in a GitHub Check Run, and a per-test ranking model that cut median time-to-first-failure by [X]% (APFD [X] vs [Y] default order).
- Deployed on AWS with Terraform (ECS Fargate, RDS, ElastiCache, S3) and a GitHub Actions CI/CD pipeline; fail-open scoring at p95 [X] ms, MLflow shadow deployments, and PSI drift monitoring with automatic retraining.
