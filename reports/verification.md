# Verification record

Verified locally on November 8, 2026 with Python 3.12 and the committed dependency lock.

| Check | Result |
| --- | --- |
| Full raw dataset validation | 10,000 users, 10,000 videos, 100,000 interactions; keys and required values valid |
| Archive raw-data consistency | SHA-256 matches across all seven project archives |
| Full-data training | Completed; five Optuna trials; selected model and pre-watch ranker exported |
| Held-out split | 64,000 train / 16,000 validation / 20,000 test, strict time order |
| Processed formats | XZ-compressed CSV and Parquet have the same 100,000 rows and 173 columns |
| Artifact integrity | Classifier, ranker and feature pipeline SHA-256 verified on loading |
| Automated tests | 32 passed: validation, temporal features, API, Redis/fallback and UI |
| Code checks | Ruff lint and formatting checks passed |
| Actual local HTTP services | API health/prediction/batch/recommendations/metrics, Streamlit page and MLflow experiment query passed |
| MLflow run database | Completed runs included; external MLflow telemetry disabled |
| Container deployment | Reproducible Compose workflow checks the real five-service stack in GitHub Actions; see the run for this commit |

Local command: `python scripts/verify_local.py`. Container command: `python scripts/smoke_test.py --stack` after `docker compose up --build -d --wait`.

The development host has no Docker daemon or GPU. The CUDA training image is provided but GPU execution is not claimed. Test scores and dataset limitations are recorded in the experiment and data exploration reports.
