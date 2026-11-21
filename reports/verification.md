# Verification record (v2, real data)

Date: 2026-11-20. Environment: Linux, 2 vCPU, 7 GB RAM, Python 3.12.3 with `requirements-lock.txt`
(lightgbm 4.7.0, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.5.4, optuna 4.9.0, mlflow 3.18.0,
fastapi 0.162.2, uvicorn 0.54.0, streamlit 1.67.0), Redis 7.0.15, Prometheus 3.13.5.

## Data

KuaiRand-Pure source files (MD5):

| File | MD5 |
| --- | --- |
| log_standard_4_08_to_4_21_pure.csv | f7396299d9e0d303adca8e795066b3f2 |
| log_standard_4_22_to_5_08_pure.csv | 40ebc8027d9136a8b8574ff58d7e66ab |
| user_features_pure.csv | c7ad8218429f511868c6335970555de5 |
| video_features_basic_pure.csv | 628641f5eec078fcd9d50ce938cb8dc5 |

The two logs hold 1,141,112 + 295,497 = 1,436,609 rows, the published KuaiRand-Pure count.
`python scripts/prepare_kuairand.py` produced the three CSVs in `data/` (SHA-256 in
`data/checksums.sha256`); `python -m src.data_validation` validates them.

## Training and evaluation

`python -m src.train` (classifier: 5 Optuna trials; ranker: 12 Optuna trials, 10 training
negatives per query) ran in 26 minutes. All model and baseline choices used validation only; the
test split was scored once. Results: `reports/ranking-report.md`, `reports/ranking_results.json`,
`reports/training_results.json`, MLflow experiments `youtube-shorts-ranking` and
`youtube-shorts-chronological` in `experiments/mlflow.db`.

Process note: a development run whose features froze every history at the training cutoff showed
time-relative features (recency, 24-hour plays) far outside their training range during the
17-day test window, and logistic regression collapsed from validation to test. The protocol was
changed for every model to read the play log point-in-time (each request sees only earlier plays;
learned inputs still use training plays only), and the ranker gained a leave-user-out co-watch
feature after it trailed ItemKNN on validation. The frozen-history variant is still reported in the
ranking report's sensitivity section for the final models.

## Automated checks

- `pytest -q`: 53 passed (feature correctness against brute force, point-in-time and future-label
  leakage, leave-user-out co-watch, NDCG against scikit-learn, candidate rules, paired bootstrap,
  dataset build, API, Redis cache and fallback, rate limits, metrics, Streamlit UI).
- `ruff check .` and `ruff format --check .`: clean.
- `docker compose config --quiet`: valid. `promtool check rules monitoring/rules.yml`: 6 rules.

## Service stack

Docker Hub was not reachable from this environment, so images were not built here; GitHub
Actions builds and starts the Compose stack. The same five services were run natively with the
Compose settings (API with real Redis, MLflow on `experiments/mlflow.db`, Streamlit, Prometheus
with `monitoring/prometheus.yml` and `monitoring/rules.yml`):

- `python scripts/smoke_test.py --stack`: passed (Redis cache backend, cached repeat requests,
  cold-start user, metrics, Streamlit health and page, MLflow, Prometheus rules).
- Prometheus: target `recommendation-api` up; all 6 rules healthy; recorded series for per-route
  p95 latency and per-endpoint cache hit ratio present.
- Streamlit rendered against the live API without exceptions (model insights, recommendations).
- `python scripts/benchmark_latency.py --requests 1000 --concurrency 1 4 8`: see
  `reports/latency.md`. Client-side p95 agreed with the API's Prometheus histogram within 1 ms for
  single-client uncached `/recommend`.
