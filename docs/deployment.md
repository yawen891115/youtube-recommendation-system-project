# Deployment and operations

## Prerequisites

The regular demo needs Docker Engine / Docker Desktop with Compose v2. It uses CPU inference; no NVIDIA GPU is required. Python development and the committed artifact versions are tested on Python 3.12. Exact packages, including test tools, are pinned in `requirements-lock.txt`; the smaller `requirements.txt` expresses supported dependency ranges.

## Service stack

Run from the repository root:

```bash
docker compose config
docker compose up --build -d --wait --wait-timeout 240
python scripts/smoke_test.py --stack
docker compose ps
```

The API loads the committed models through a read-only mount. Redis starts before the API. The UI, MLflow and Prometheus wait for API readiness. MLflow's named volume is initialized from the image's included `experiments/mlflow.db` on its first creation. Subsequent container restarts retain the existing volume; they do not overwrite its database with a newly committed one.

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_DIR` | `models` | Trusted model artifact directory |
| `REDIS_URL` | unset for local Python; `redis://redis:6379/0` in Compose | Cache / rate-limit store |
| `REQUESTS_PER_MINUTE` | `120` | Per-client-IP limit for prediction and recommendation requests |
| `API_BASE_URL` | `http://127.0.0.1:8000` | UI's API URL; Compose sets `http://api:8000` |
| Prediction cache TTL | 60 seconds | Model-version and effective-request keyed cache |

All published Compose ports bind to `127.0.0.1`. Redis stays on the Compose network and has no published host port. The application does not trust arbitrary forwarded IP headers. Add authentication and an appropriately configured proxy before any public deployment.

The `/health` endpoint returns HTTP 503 when artifacts are absent, corrupt or incompatible. Predictions also return HTTP 503 in that state; they never fall back to an untrained placeholder model. Redis failure falls back to bounded process-local state. This fallback is for a single process and does not provide a distributed rate limit.

Prometheus scrapes `/metrics` every 15 seconds. Request labels use bounded route names. The Streamlit dashboard's `/monitoring` history holds the most recent 1,000 application requests in memory; it is not a durable analytics database. Standard HTTP logs report route, status and latency.

## Train and inspect experiments

```bash
python -m src.train --trials 5
MLFLOW_DISABLE_TELEMETRY=true mlflow server --backend-store-uri sqlite:///experiments/mlflow.db --host 127.0.0.1 --port 5000
```

MLflow telemetry is disabled in the project and both containers. Experiment parameters and metrics stay in the configured local SQLite store.

The committed database includes actual runs. A human-readable export is provided in `reports/mlflow_runs.json` and `reports/optuna_trials.csv`. Training rewrites the model bundle and processed dataset. Restart the API after replacing its model bundle; its cache key includes the classifier, ranker and feature-pipeline version.

The assignment's optional CUDA-based training environment is supplied separately:

```bash
docker build -f Dockerfile.train -t youtube-shorts-training .
docker run --rm --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/experiments:/app/experiments" \
  youtube-shorts-training
```

This image uses NVIDIA CUDA 12.6.3 on Ubuntu 24.04 with Python 3.12. GPU access requires the NVIDIA Container Toolkit and a compatible host driver. The bundled LightGBM wheel and training configuration use CPU; a CUDA base image alone does not make training GPU-accelerated. GPU performance is not claimed or benchmarked. The standard CPU serving stack is the automated deployment verification target.

## Verification and recovery

The GitHub Actions workflow runs validation, feature/API/UI tests and artifact checks, then builds and starts the real Compose stack. Its smoke script checks the API, Redis-backed cache, recommendations, metrics, Streamlit page/health endpoint, MLflow and Prometheus. Review the workflow run associated with the commit being reviewed.

For diagnostics:

```bash
docker compose logs --tail=100 api streamlit mlflow redis prometheus
curl --fail http://localhost:8000/health
```

Use `docker compose down` to stop services without deleting named volumes. Only use `docker compose down -v` when intentionally resetting local runtime state. No cloud service, API key or paid model endpoint is needed.
