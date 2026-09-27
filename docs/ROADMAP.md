# Preflight — coverage map

Every item in SPEC.md mapped to the week that builds it. Items marked **(gap)** are in the
SPEC but were not named in the SPEC §9 build plan; they are assigned here so nothing is lost.
Tick items only when they are built *and* tested.

## §1 Product features
| # | Feature | Week |
|---|---|---|
| 1 | Risk check (calibrated P(fail), bands, per-repo thresholds) | W3 model · W4 Check Run |
| 2 | Plain-language reasons (SHAP → templates) | W3 |
| 3 | What-if fixes (re-score edited actionable features) | W3 |
| 4 | Fail-fast test order (per-test model, APFD, pytest plugin) | W3 model · W4 plugin |
| 5 | Adaptive CI plan (lint+affected / full matrix; main always full) **(gap)** | W4 (scorer returns plan) · W5 (own workflow consumes it) |
| 6 | Flaky test registry + quarantine suggestion **(gap)** | W1 flaky labels · W2 per-test flaky_score · W6 UI |
| 7 | Pre-push CLI `preflight check` | W6 |
| 8 | Dashboard (hotspot treemap, CI minutes saved, calibration, drift) | W6 |
| 9 | Feedback votes **(gap)** + weekly Slack digest | W4 votes (Check Run action) · W6 digest |

## §2 Architecture
| Item | Week |
|---|---|
| Ingest API: HMAC verify, dedup by delivery ID, → Redis Streams | W1 |
| Feature builder (point-in-time) | W2 offline · W4 online |
| Risk scorer (LightGBM + SHAP + conformal) → Check Run | W3 · W4 |
| Labeler (taxonomy) on workflow_run completed | W1 batch · W4 live |
| Feature store: S3 Parquet + Postgres `features` | W2 |
| Trainer: temporal CV, calibration, threshold | W2 · W3 |
| MLflow registry shadow → champion | W5 |
| Monitor: PSI drift, rolling PR-AUC → retrain trigger | W5 |
| Nightly batch: aggregates → Redis online store **(gap)** | W4 |

## §3 Data sources
| Item | Week |
|---|---|
| GitHub Actions API, 20–30 OSS repos (runs, jobs, reruns, logs, diffs) | W1 |
| JUnit XML / coverage.xml artifacts | W1 (where available) |
| TravisTorrent benchmark / global pretraining **(gap)** | W2 load · W3 benchmark |
| Dogfooding (Preflight's own CI) | W1 CI workflow · W5 scores own PRs |

## §4 Failure taxonomy
| Item | Week |
|---|---|
| Rules: code_defect / flaky / infra / config / cancelled | W1 (v0) |
| Hand-label ~300 logs, measure precision | W1 export · kd labels · W2 measure |
| TF-IDF + logistic-regression log classifier **(gap)** | W2 |

## §5 Feature families (all point-in-time)
Size & spread · Change type · File history · Co-change · Test signal · Context · Author — **W2**
(Bayesian smoothing, co-change mining, leakage test included.)

## §6 Data model
| Table | Week |
|---|---|
| repos, commits, file_changes, builds, test_results | W1 |
| features | W2 |
| predictions, feedback | W4 |
| model_runs | W3 (written by trainer) · W5 (registry status) |
| Row-level security by installation_id **(gap)** | W5 |

## §8 Production checklist
| Area | Item | Week |
|---|---|---|
| Reliability | fail-open (>2 s / error → skipped check) | W4 |
| | idempotent webhooks | W1 |
| | retries + DLQ | W1 client retries · W4 stream DLQ |
| | ETag conditional requests + token bucket | W1 |
| | resumable backfill | W1 |
| MLOps | shared feature defs + train/serve parity test | W2 defs · W4 parity test |
| | MLflow tags (window, feature hash, metrics) | W3 log · W5 registry |
| | 7-day shadow mode; one-command rollback | W5 |
| | PSI drift; weekly + triggered retrain | W5 |
| | Pandera validation | W2 |
| Security | least-privilege App; HMAC-SHA256 | W1 |
| | no source stored; author IDs hashed | W1 (enforced in backfill) |
| | short-lived installation tokens | W4 |
| | Secrets Manager | W5 |
| | RLS | W5 |
| | blame-free mode (author features off) | W3 ablation · W4 flag |
| Observability | JSON logs keyed by delivery ID | W1 |
| | OpenTelemetry traces; Prometheus metrics; Grafana alerts | W5 |
| Testing | recorded webhook fixtures | W1 |
| | leakage test | W2 |
| | behavioural tests (adding tests never raises risk; whitespace diff stays low) | W3 |
| | Locust 50 webhooks/s | W6 |
| Delivery | Terraform (ECS Fargate, RDS, ElastiCache, S3, Secrets Manager) | W5 |
| | GH Actions lint → test → build → ECR → staging → smoke → prod | W1 lint+test · W5 full |
| | Docker Compose + webhook replayer | W1 |

## §9 Modelling (from SPEC non-negotiables)
| Item | Week |
|---|---|
| Baselines: always-pass, files-changed > N, logistic regression | W2 |
| LightGBM monotonic, class weights, SMOTE ablation | W3 |
| Isotonic calibration, ECE/Brier, reliability curve | W3 |
| Cost-based per-repo threshold | W3 |
| Conformal abstention (MAPIE) | W3 |
| Global → per-repo model after ~500 builds | W3 |
| Feature-family ablations | W3 |
| p95 latency < 300 ms | W4 measure · W6 load test |

## §10 Resume numbers
Filled only from real runs, in W6 (README results table).
