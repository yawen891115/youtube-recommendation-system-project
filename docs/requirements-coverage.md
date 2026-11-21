# Assignment coverage

The exact supplied English specification is preserved in [assignment-original.en.md](assignment-original.en.md). The table below maps its requirements to the consolidated implementation and verification. This is a completion record, not a claim of a course grade.

**Dataset change (v2).** The supplied tables were synthetic: which video a user watched was uniformly random and the only pre-watch signal was a planted category-preference match, so no ranker could beat a simple rule. Version 2 keeps the same schema-driven pipeline but uses a reproducible 100,000-play subset of the real KuaiRand-Pure logs (Kuaishou, CC BY-SA 4.0); see [data/README.md](../data/README.md). User demographics (age, gender, declared category) do not exist in that source and were replaced by its real user attributes.

| Assignment requirement | Implementation / deliverable | Verification |
| --- | --- | --- |
| Load users, videos and interactions | `src/data_validation.py`; KuaiRand-Pure subset built by `scripts/prepare_kuairand.py` | Schema, foreign-key and duplicate tests; full-data validation |
| Integrity, missing cells and anomalies | Pandera schemas; `describe_data`; dataset report | Invalid activity level, duration, infinite watch and unknown IDs rejected; plays before upload counted |
| Data scale / distributions | `reports/data-exploration.md`, `training_results.json`, plot | Counts and distributions generated from actual raw files |
| Calendar, recency and day-period features | `FeatureEngineer.transform` | Timestamp-based transformation tests |
| Historical user watch, activity, likes and shares | Prefix history aggregates | Current/future label changes cannot affect prior features |
| Session ID, length and duration | 30-minute gap sessions; length/duration as observed so far | Session boundary and prior-history tests |
| Video watch, popularity, engagement and freshness | Historical interaction counts/rates and catalog upload age | Shared feature tests; negative ages clipped and reported |
| Completion, normalization and one-hot encoding | Training-fitted scalers and explicit category encoder | Finite features and unknown-ID tests |
| SVD embeddings, 50 dimensions | Sparse interaction matrix; 50 dimensions per user and video | Warmup embedding mask; saved embedding artifacts |
| Advanced interactions | Hour/weekend, watch/popularity, recent activity, category diversity | Included in saved feature schema |
| Processed CSV and feature artifacts | `processed_interactions.csv.xz`, Parquet, scalers/encoders/embeddings | Full row counts; CSV decompression command in README |
| 64% / 16% / 20%, seed 42 | Strict chronological split with tied timestamps kept together | Partition size and timestamp tests |
| Class proportions / stratification | Per-split prevalence report and training class weighting | Intentional resolution described below |
| Logistic Regression, Random Forest, Gradient Boosting | `src/train.py` baseline comparison | Actual validation accuracy, AUC and other metrics in reports |
| LightGBM, early stopping, imbalance | Validation-only early stopping; `is_unbalance=True` | Full dataset training run |
| Optuna search and best parameters | Seeded study over leaves, learning rate, feature fraction and leaf sample count | Trial CSV, best parameters and validation AUC |
| Saved best model and feature columns | `models/best_model_*.pkl`, `feature_columns.pkl` | SHA-256 checked at API startup |
| MLflow tracking | `experiments/mlflow.db`, JSON export | Recorded runs, parameters and metrics |
| Pre-watch ranking (v2 addition) | `src/ranking.py`, `src/train_ranker.py`: candidate lists, LightGBM LambdaRank, six baselines, NDCG@10 / HR@10 / MRR | Metric checked against scikit-learn; paired bootstrap test; leakage and point-in-time tests |
| Health, model info, single and batch APIs | `src/api.py` | Success, consistency and model-unavailable tests |
| Pydantic request/response validation | Strict IDs, finite watch time, timezone, bounds | Invalid input, missing/oversized batch tests |
| Redis caching and rate limit | `src/cache.py`; model-aware keys and bounded fallback | FakeRedis behavior tests and real Redis in Compose smoke |
| Async processing | Async endpoints; CPU prediction offloaded to thread pool | HTTP endpoint tests and smoke script |
| Prometheus, logs, response time | Request counters, latency histograms, bounded monitoring history | Metrics and monitoring assertions |
| Single UI prediction and gauge | English Streamlit single page | UI render, submit and API error tests |
| CSV batch, downloads and histogram | English Streamlit batch page; 256-row requests | API batch consistency and limits; example CSV supplied |
| Dashboard trends, errors, latency, importance | Monitoring page backed by API data | UI rendering test and metrics endpoint checks |
| CUDA Dockerfile | `Dockerfile.train` | Configuration provided; GPU execution not benchmarked |
| API / Redis / MLflow Compose and volumes | `docker-compose.yml`, plus UI and Prometheus | Actual Compose workflow, health checks and smoke script |
| Endpoint and frontend availability | `scripts/smoke_test.py --stack` | GitHub Actions Compose job; local Python smoke |
| Code, EDA, experiment report and deployment guide | Repository, `reports/`, `docs/deployment.md` | Included, linked and reproducible |

## Deliberate resolutions and limitations

**Chronology and stratification.** The assignment asks for strict chronological splits and stratified sampling simultaneously. Equal class ratios generally cannot be forced while preserving every row and strict time boundaries. This implementation keeps the time split and all rows, reports prevalence for each partition, and uses imbalance handling during training. It does not randomly move future events into earlier partitions.

**Feature availability.** User/video statistics query only plays strictly before the prediction time. Learned inputs (priors, scaler, SVD basis, co-watch counts) use training plays only; the history that features read is the play log up to each request, as an online feature store would serve it (the ranking report also shows results with histories frozen at the training cutoff). SVD is fitted on the first 10% of training history, with zero embeddings for those warmup rows. Static user/video catalog metadata is assumed to be available; the supplied snapshot does not allow historical metadata availability to be proven. Aggregate catalog `view_count` is not used as a historical feature because its as-of date is unknown.

**Prediction versus recommendation.** The assignment's request includes observed `watch_time`, so `/predict` estimates engagement after observation. The additional `/recommend` model removes watch time, normalized watch time, completion and watch/popularity ratio before ranking. Reported offline AUC is not an online recommendation uplift measurement.

**Output size.** The required CSV is included with lossless XZ compression and a Parquet copy. Decompressing the supplied `.csv.xz` produces the CSV without retraining. The stale 196 MB derived file from the archives is superseded by features generated from the current code.

**Deployment verification.** The v2 development environment could not pull images from Docker Hub and has no NVIDIA GPU. All five services (API, Redis, MLflow, Prometheus, Streamlit) were run natively with the same versions and configuration, and `docker compose config` and `promtool check` validate the Compose and Prometheus files; the real CPU Compose stack is exercised in GitHub Actions. The separate CUDA training image is provided for the assignment's GPU scenario, with CPU training as the documented default.
