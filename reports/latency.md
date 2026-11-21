# Latency benchmark

Measured 2026-11-20T23:53:28.587498+00:00 against `http://127.0.0.1:8000` (five services run natively (API: 1 uvicorn worker, real Redis) on a 2-vCPU sandbox; load generator on the same host); cache backend `redis`. Closed-loop clients; 1000 requests per phase; 10% of /recommend requests are new (cold-start) users. Client-side latency includes HTTP and JSON; server p95 comes from the API's Prometheus histogram over the same interval (bucket-interpolated).

| Phase | Concurrency | p50 ms | p95 ms | p99 ms | Throughput req/s | Cache hits | Errors | Server p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recommend_cache_miss | 1 | 15.5 | 19.7 | 23.0 | 62 | 0% | 0.0% | 19.5 |
| recommend_cache_hit | 1 | 2.1 | 2.6 | 3.9 | 436 | 100% | 0.0% | 2.4 |
| predict_cache_miss | 1 | 7.2 | 9.6 | 10.2 | 132 | 0% | 0.0% | 9.8 |
| recommend_cache_miss | 4 | 44.3 | 62.9 | 76.1 | 88 | 0% | 0.0% | 69.1 |
| recommend_cache_hit | 4 | 5.5 | 6.4 | 7.9 | 617 | 100% | 0.0% | 4.9 |
| predict_cache_miss | 4 | 28.7 | 39.8 | 46.3 | 134 | 0% | 0.0% | 34.4 |
| recommend_cache_miss | 8 | 98.1 | 141.1 | 166.7 | 79 | 0% | 0.0% | 144.9 |
| recommend_cache_hit | 8 | 11.0 | 16.1 | 21.7 | 568 | 100% | 0.0% | 12.1 |
| predict_cache_miss | 8 | 58.6 | 80.3 | 101.7 | 127 | 0% | 0.0% | 73.8 |
