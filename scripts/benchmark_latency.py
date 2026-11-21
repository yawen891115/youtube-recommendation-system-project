"""Load-test a running API and report client-observed latency percentiles.

Phases (closed loop: each worker sends its next request when the previous one returns):
  * cold    /recommend, every request a distinct user -> cache miss -> retrieval + ranking
  * warm    the same /recommend requests replayed     -> Redis cache hit
  * predict /predict with distinct inputs             -> cache miss -> like classifier

For each phase and concurrency level the script records p50/p95/p99, throughput and errors,
and cross-checks p95 against the API's own Prometheus histogram (server-side, same interval).

Start the API with a high request limit and a cache TTL longer than one phase, e.g.
    REQUESTS_PER_MINUTE=1000000 CACHE_TTL_SECONDS=600 REDIS_URL=redis://localhost:6379/0 \
        uvicorn src.api:app --port 8000
then run
    python scripts/benchmark_latency.py --requests 1000 --concurrency 1 4 8
"""

import argparse
import json
import platform
import queue
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

CATEGORIES = ["Comedy", "Education", "Fashion", "Food", "Gaming", "Music", "Sports", "Travel"]
BUCKET = re.compile(
    r'recommendation_http_duration_seconds_bucket\{le="([^"]+)",route="([^"]+)"\} ([0-9.e+]+)'
)


def histogram(api, route):
    text = httpx.get(api + "/metrics", timeout=30).text
    rows = [(float(le), float(count)) for le, path, count in BUCKET.findall(text) if path == route]
    return sorted(rows)


def server_quantile(before, after, quantile):
    """Prometheus-style histogram_quantile over the counts added between two scrapes."""
    previous = dict(before)
    buckets = [(le, count - previous.get(le, 0.0)) for le, count in after]
    total = buckets[-1][1] if buckets else 0
    if not total:
        return None
    rank, lower, below = quantile * total, 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if np.isinf(le):
                return lower * 1000
            return (lower + (le - lower) * (rank - below) / max(count - below, 1e-12)) * 1000
        lower, below = le, count
    return None


def run_phase(api, path, bodies, concurrency):
    jobs = queue.Queue()
    for body in bodies:
        jobs.put(body)
    latencies, statuses, cached = [], [], []
    lock = threading.Lock()

    def worker():
        with httpx.Client(base_url=api, timeout=30) as client:
            while True:
                try:
                    body = jobs.get_nowait()
                except queue.Empty:
                    return
                start = time.perf_counter()
                response = client.post(path, json=body)
                elapsed = (time.perf_counter() - start) * 1000
                hit = response.json().get("cached") if response.status_code == 200 else None
                with lock:
                    latencies.append(elapsed)
                    statuses.append(response.status_code)
                    cached.append(hit)

    route = "/recommend" if path == "/recommend" else "/predict"
    before = histogram(api, route)
    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    after = histogram(api, route)
    values = np.array(latencies)
    return {
        "requests": len(values),
        "concurrency": concurrency,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "max_ms": float(values.max()),
        "throughput_rps": len(values) / wall,
        "error_rate": float(np.mean([status != 200 for status in statuses])),
        "cache_hit_rate": float(np.mean([bool(hit) for hit in cached])),
        "server_p95_ms": server_quantile(before, after, 0.95),
    }


def build_requests(users, count, start, seed, cold_start_share):
    """Distinct (user, time) requests; a share use never-seen user IDs (cold start)."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(users, size=count, replace=False)
    bodies = []
    for index, user in enumerate(chosen):
        if rng.random() < cold_start_share:
            user = f"user_{9_000_000 + seed * 100_000 + index}"
        bodies.append({"user_id": str(user), "top_k": 10, "timestamp": start.isoformat()})
    return bodies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--cold-start-share", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", default="reports")
    parser.add_argument("--label", default="local")
    args = parser.parse_args()

    health = httpx.get(args.api + "/health", timeout=30).json()
    if not health.get("model_loaded"):
        raise SystemExit(f"API not ready: {health}")
    users = pd.read_csv(Path(args.data_dir) / "users.csv").user_id.to_numpy()
    videos = pd.read_csv(Path(args.data_dir) / "videos.csv").video_id.to_numpy()
    # The bundled data ends on 2022-05-08; requests replay times just after its last play.
    start = datetime(2022, 5, 8, 16, tzinfo=timezone.utc)
    # Warm-up (not measured): load code paths, open connections.
    warmup = build_requests(users, args.warmup, start - timedelta(days=1), 999, 0.0)
    run_phase(args.api, "/recommend", warmup, 1)

    results = []
    for offset, concurrency in enumerate(args.concurrency):
        # A distinct request time per level keeps every cold request a cache miss.
        when = start + timedelta(minutes=offset)
        bodies = build_requests(users, args.requests, when, offset, args.cold_start_share)
        cold = run_phase(args.api, "/recommend", bodies, concurrency)
        warm = run_phase(args.api, "/recommend", bodies, concurrency)
        rng = np.random.default_rng(100 + offset)
        predict = [
            {
                "user_id": str(user),
                "video_id": str(video),
                "watch_time": float(rng.uniform(1, 60)),
                "timestamp": when.isoformat(),
            }
            for user, video in zip(
                rng.choice(users, args.requests), rng.choice(videos, args.requests)
            )
        ]
        scored = run_phase(args.api, "/predict", predict, concurrency)
        for phase, row in [
            ("recommend_cache_miss", cold),
            ("recommend_cache_hit", warm),
            ("predict_cache_miss", scored),
        ]:
            results.append({"phase": phase, **row})
            print(
                f"{phase:22s} c={concurrency:<3d} p50 {row['p50_ms']:7.2f} ms  p95 "
                f"{row['p95_ms']:7.2f} ms  p99 {row['p99_ms']:7.2f} ms  "
                f"{row['throughput_rps']:7.1f} req/s  hits {row['cache_hit_rate']:.0%}  "
                f"errors {row['error_rate']:.1%}  server p95 {row['server_p95_ms'] or 0:.1f} ms"
            )

    report = {
        "label": args.label,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "api": args.api,
        "cache_backend": health.get("cache_backend"),
        "requests_per_phase": args.requests,
        "cold_start_share": args.cold_start_share,
        "top_k": 10,
        "candidates_ranked_per_request": "up to 300",
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "results": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Latency benchmark",
        "",
        f"Measured {report['measured_at']} against `{args.api}` ({args.label}); cache backend "
        f"`{report['cache_backend']}`. Closed-loop clients; {args.requests} requests per phase; "
        f"{args.cold_start_share:.0%} of /recommend requests are new (cold-start) users. "
        "Client-side latency includes HTTP and JSON; server p95 comes from the API's Prometheus "
        "histogram over the same interval (bucket-interpolated).",
        "",
        "| Phase | Concurrency | p50 ms | p95 ms | p99 ms | Throughput req/s | Cache hits | Errors | Server p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['phase']} | {row['concurrency']} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | "
            f"{row['p99_ms']:.1f} | {row['throughput_rps']:.0f} | {row['cache_hit_rate']:.0%} | "
            f"{row['error_rate']:.1%} | {row['server_p95_ms'] or 0:.1f} |"
        )
    lines.append("")
    (output / "latency.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
