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
from .inference import RecommendationService

logger = logging.getLogger(__name__)
UserID = Annotated[str, Field(pattern=r"^user_\d+$", max_length=64)]
VideoID = Annotated[str, Field(pattern=r"^video_\d+$", max_length=64)]


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
        # The default represents the current UTC minute, making short-lived cache keys stable.
        when = self.timestamp or datetime.now(timezone.utc).replace(second=0, microsecond=0)
        when = when.astimezone(timezone.utc)
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


def create_app(service=None, cache=None, model_dir=None, rate_limit=None):
    cache = cache or CacheStore(os.getenv("REDIS_URL"))
    limit = rate_limit if rate_limit is not None else int(os.getenv("REQUESTS_PER_MINUTE", "120"))
    registry = CollectorRegistry()
    requests_metric = Counter(
        "recommendation_http_requests_total",
        "HTTP requests",
        ["route", "status"],
        registry=registry,
    )
    latency = Histogram(
        "recommendation_http_duration_seconds", "HTTP latency", ["route"], registry=registry
    )
    cache_hits = Counter(
        "recommendation_cache_hits_total", "Prediction cache hits", registry=registry
    )
    predictions_metric = Counter(
        "recommendation_predictions_total", "Instances scored", registry=registry
    )
    events = deque(maxlen=1000)

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        if app.state.service is None:
            try:
                app.state.service = await run_in_threadpool(
                    RecommendationService, model_dir or os.getenv("MODEL_DIR", "models")
                )
            except Exception:
                logger.exception("Model loading failed; prediction endpoints will return 503")
        await cache.ping()
        yield
        await cache.close()

    app = FastAPI(title="YouTube Shorts Recommendation API", version="1.0.0", lifespan=lifespan)

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
            if route not in ["/metrics", "/monitoring", "/health"]:
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
        return {
            key: loaded.metadata[key]
            for key in [
                "model_version",
                "model_name",
                "feature_columns",
                "training_cutoff",
                "embedding_cutoff",
                "feature_importance",
                "validation_metrics",
                "test_metrics",
            ]
        } | {
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
                loaded.predict, [records[index] for index in missing]
            )
            for index, value in zip(missing, computed):
                values[index] = value
                await cache.set(keys[index], value)
        predictions_metric.inc(len(records))
        cache_hits.inc(sum(hits))
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
        when = payload.timestamp or datetime.now(timezone.utc)
        try:
            videos = await run_in_threadpool(loaded.recommend, payload.user_id, payload.top_k, when)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "user_id": payload.user_id,
            "model_version": loaded.version,
            "recommendations": videos,
            "candidate_limit": 300,
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/monitoring")
    async def monitoring():
        rows = list(events)
        return {
            "window_requests": len(rows),
            "error_rate": sum(row["status"] >= 400 for row in rows) / max(len(rows), 1),
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / max(len(rows), 1),
            "cache_backend": cache.backend,
            "events": rows,
        }

    return app


app = create_app()
