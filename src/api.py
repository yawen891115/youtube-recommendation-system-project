"""Validated asynchronous prediction, recommendation, and monitoring endpoints."""

import hashlib
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .cache import CacheStore
from .inference import CANDIDATE_LIMIT, RecommendationService

logger = logging.getLogger(__name__)
UserID = Annotated[str, Field(pattern=r"^user_\d+$", max_length=64)]
VideoID = Annotated[str, Field(pattern=r"^video_\d+$", max_length=64)]
# Fine buckets between 5 and 100 ms keep histogram_quantile(0.95, ...) precise.
LATENCY_BUCKETS = (
    0.0025,
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.05,
    0.075,
    0.1,
    0.15,
    0.25,
    0.5,
    1.0,
    2.5,
)
UNTRACKED_ROUTES = {"/metrics", "/monitoring", "/health"}


def current_minute():
    """Default request time. REPLAY_TIME pins it inside a historical dataset (demo replay);
    otherwise the current UTC minute keeps short-lived cache keys stable within a minute."""
    replay = os.getenv("REPLAY_TIME")
    if replay:
        return datetime.fromisoformat(replay.replace("Z", "+00:00")).astimezone(timezone.utc)
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    user_id: UserID
    video_id: VideoID
    watch_time: float = Field(ge=0, le=3600)
    hour_of_day: int | None = Field(default=None, ge=0, le=23)
    timestamp: AwareDatetime | None = None

    @model_validator(mode="after")
    def consistent_hour(self):
        if self.timestamp is not None and self.hour_of_day is not None:
            if self.timestamp.astimezone(timezone.utc).hour != self.hour_of_day:
                raise ValueError("hour_of_day must match timestamp in UTC")
        return self

    def record(self):
        when = (self.timestamp or current_minute()).astimezone(timezone.utc)
        if self.hour_of_day is not None:
            when = when.replace(hour=self.hour_of_day)
        return {
            "user_id": self.user_id,
            "video_id": self.video_id,
            "watch_time": self.watch_time,
            "timestamp": when.isoformat(),
        }


class PredictionResponse(BaseModel):
    probability: float
    confidence: str
    cold_start: bool
    model_version: str
    response_time_ms: float
    timestamp: str
    cached: bool
    confidence_basis: str = "uncalibrated distance from 0.5"


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PredictionRequest] = Field(min_length=1, max_length=256)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: UserID
    top_k: int = Field(default=10, ge=1, le=50)
    timestamp: AwareDatetime | None = None


def create_app(service=None, cache=None, model_dir=None, rate_limit=None, cache_ttl=None):
    cache = cache or CacheStore(os.getenv("REDIS_URL"))
    limit = rate_limit if rate_limit is not None else int(os.getenv("REQUESTS_PER_MINUTE", "120"))
    ttl = cache_ttl if cache_ttl is not None else int(os.getenv("CACHE_TTL_SECONDS", "60"))
    registry = CollectorRegistry()
    requests_metric = Counter(
        "recommendation_http_requests_total",
        "HTTP requests",
        ["route", "status"],
        registry=registry,
    )
    latency = Histogram(
        "recommendation_http_duration_seconds",
        "End-to-end HTTP latency inside the API process",
        ["route"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    errors = Counter(
        "recommendation_http_errors_total",
        "HTTP responses with status >= 400",
        ["route", "status_class"],
        registry=registry,
    )
    cache_requests = Counter(
        "recommendation_cache_requests_total",
        "Response cache lookups",
        ["endpoint", "result"],
        registry=registry,
    )
    model_seconds = Histogram(
        "recommendation_model_seconds",
        "Feature building plus model scoring time",
        ["model"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    predictions_metric = Counter(
        "recommendation_predictions_total", "Instances scored", registry=registry
    )
    recommendations_metric = Counter(
        "recommendation_requests_total",
        "Recommendation lists served",
        ["user_type"],
        registry=registry,
    )
    events = deque(maxlen=1000)
    cache_counts = {"hit": 0, "miss": 0}

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        if app.state.service is None:
            try:
                app.state.service = await run_in_threadpool(
                    RecommendationService,
                    model_dir or os.getenv("MODEL_DIR", "models"),
                    int(os.getenv("CANDIDATE_LIMIT", str(CANDIDATE_LIMIT))),
                )
            except Exception:
                logger.exception("Model loading failed; prediction endpoints will return 503")
        await cache.ping()
        yield
        await cache.close()

    app = FastAPI(title="YouTube Shorts Recommendation API", version="2.0.0", lifespan=lifespan)

    def get_service():
        loaded = app.state.service
        if loaded is None:
            raise HTTPException(503, "Model unavailable. Run python -m src.train first.")
        return loaded

    async def check_limit(request):
        client = request.client.host if request.client else "unknown"
        if not await cache.allow(client, limit):
            raise HTTPException(
                429,
                "Request limit exceeded",
                headers={"Retry-After": str(60 - int(time.time()) % 60)},
            )

    def count_cache(endpoint, hit):
        result = "hit" if hit else "miss"
        cache_requests.labels(endpoint, result).inc()
        cache_counts[result] += 1

    def timed(model, function, *args):
        start = time.perf_counter()
        try:
            return function(*args)
        finally:
            model_seconds.labels(model).observe(time.perf_counter() - start)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            elapsed = time.perf_counter() - start
            requests_metric.labels(route, str(status)).inc()
            latency.labels(route).observe(elapsed)
            if status >= 400:
                errors.labels(route, f"{status // 100}xx").inc()
            if route not in UNTRACKED_ROUTES:
                events.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "route": route,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 3),
                    }
                )
            logger.info("request route=%s status=%s latency_ms=%.3f", route, status, elapsed * 1000)

    @app.get("/health")
    async def health(response: Response):
        ready = app.state.service is not None
        if not ready:
            response.status_code = 503
        return {
            "status": "ok" if ready else "not_ready",
            "model_loaded": ready,
            "cache_backend": cache.backend,
        }

    @app.get("/model/info")
    async def model_info():
        loaded = get_service()
        metadata = loaded.metadata
        return {
            key: metadata.get(key)
            for key in [
                "model_version",
                "model_name",
                "feature_columns",
                "training_cutoff",
                "embedding_cutoff",
                "feature_importance",
                "validation_metrics",
                "test_metrics",
                "ranking_model",
                "ranking_benchmark",
                "ranking_ablations",
                "ranking_feature_importance",
                "ranking_feature_group_importance",
            ]
        } | {
            "ranking_feature_count": len(metadata["ranking_feature_columns"]),
            "candidate_limit": loaded.candidate_limit,
            "confidence_basis": "heuristic, uncalibrated",
            "ranking_excludes_observed_watch_features": True,
        }

    async def score(instances):
        loaded = get_service()
        start = time.perf_counter()
        records = [instance.record() for instance in instances]
        keys = [
            "prediction:"
            + hashlib.sha256(
                json.dumps([loaded.version, record], sort_keys=True).encode()
            ).hexdigest()
            for record in records
        ]
        values = [await cache.get(key) for key in keys]
        hits = [value is not None for value in values]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            computed = await run_in_threadpool(
                timed, "classifier", loaded.predict, [records[index] for index in missing]
            )
            for index, value in zip(missing, computed):
                values[index] = value
                await cache.set(keys[index], value, ttl=ttl)
        for hit in hits:
            count_cache("predict", hit)
        predictions_metric.inc(len(records))
        elapsed = (time.perf_counter() - start) * 1000
        now = datetime.now(timezone.utc).isoformat()
        return [
            PredictionResponse(
                **value,
                model_version=loaded.version,
                response_time_ms=round(elapsed, 3),
                timestamp=now,
                cached=hit,
            )
            for value, hit in zip(values, hits)
        ]

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionRequest, request: Request):
        await check_limit(request)
        return (await score([payload]))[0]

    @app.post("/predict/batch")
    async def predict_batch(payload: BatchRequest, request: Request):
        await check_limit(request)
        return {"predictions": await score(payload.instances), "count": len(payload.instances)}

    @app.post("/recommend")
    async def recommend(payload: RecommendationRequest, request: Request):
        await check_limit(request)
        loaded = get_service()
        start = time.perf_counter()
        when = (payload.timestamp or current_minute()).astimezone(timezone.utc)
        key = (
            "recommend:"
            + hashlib.sha256(
                json.dumps(
                    [loaded.version, payload.user_id, payload.top_k, when.isoformat()],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        body = await cache.get(key)
        hit = body is not None
        count_cache("recommend", hit)
        if not hit:
            try:
                result = await run_in_threadpool(
                    timed, "ranker", loaded.recommend, payload.user_id, payload.top_k, when
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            user = result["user"]
            body = {
                "user_id": payload.user_id,
                "model_version": loaded.version,
                "cold_start": user["history_events"] == 0,
                "history_events": user["history_events"],
                "top_categories": user["top_categories"],
                "recommendations": result["recommendations"],
                "candidate_count": result["candidates"],
                "candidate_limit": loaded.candidate_limit,
                "request_time": when.isoformat(),
            }
            await cache.set(key, body, ttl=ttl)
        recommendations_metric.labels("cold_start" if body["cold_start"] else "known").inc()
        return body | {
            "cached": hit,
            "response_time_ms": round((time.perf_counter() - start) * 1000, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/monitoring")
    async def monitoring():
        rows = list(events)
        latencies = np.array([row["latency_ms"] for row in rows]) if rows else np.zeros(0)
        lookups = cache_counts["hit"] + cache_counts["miss"]
        return {
            "window_requests": len(rows),
            "error_rate": sum(row["status"] >= 400 for row in rows) / max(len(rows), 1),
            "mean_latency_ms": float(latencies.mean()) if rows else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if rows else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if rows else 0.0,
            "cache_hit_ratio": cache_counts["hit"] / lookups if lookups else 0.0,
            "cache_lookups": lookups,
            "cache_backend": cache.backend,
            "events": rows,
        }

    return app


app = create_app()
