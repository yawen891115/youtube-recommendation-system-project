import asyncio
from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from src.api import create_app
from src.cache import CacheStore

PAYLOAD = {
    "user_id": "user_1",
    "video_id": "video_1",
    "watch_time": 20,
    "timestamp": "2025-01-01T12:00:00Z",
}


@pytest.fixture
def client(trained_service):
    with TestClient(
        create_app(service=trained_service, cache=CacheStore(), rate_limit=1000)
    ) as test_client:
        yield test_client


def test_prediction_batch_and_monitoring(client):
    assert client.get("/health").status_code == 200
    result = client.post("/predict", json=PAYLOAD)
    assert result.status_code == 200
    data = result.json()
    assert 0 <= data["probability"] <= 1
    assert data["cached"] is False
    second = client.post("/predict", json=PAYLOAD).json()
    assert second["cached"] is True
    batch = client.post("/predict/batch", json={"instances": [PAYLOAD, PAYLOAD]}).json()
    assert batch["count"] == 2
    assert all(row["probability"] == data["probability"] for row in batch["predictions"])
    assert client.get("/monitoring").json()["window_requests"] >= 3
    assert "recommendation_predictions_total 4.0" in client.get("/metrics").text
    assert client.get("/model/info").json()["ranking_excludes_observed_watch_features"]


@pytest.mark.parametrize(
    "change",
    [
        {"user_id": "not-a-user"},
        {"video_id": "../video_1"},
        {"watch_time": -1},
        {"watch_time": "nan"},
        {"watch_time": "inf"},
        {"hour_of_day": 24},
        {"hour_of_day": 2},
        {"timestamp": "2025-01-01T12:00:00"},
        {"unexpected": 1},
    ],
)
def test_input_validation(client, change):
    assert client.post("/predict", json=PAYLOAD | change).status_code == 422


def test_batch_limit(client):
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422
    assert client.post("/predict/batch", json={"instances": [PAYLOAD] * 257}).status_code == 422


def test_unknown_ids_and_recommendations(client):
    result = client.post(
        "/predict", json=PAYLOAD | {"user_id": "user_9999", "video_id": "video_9999"}
    )
    assert result.status_code == 200
    assert result.json()["cold_start"] is True
    result = client.post("/recommend", json={"user_id": "user_1", "top_k": 3})
    assert result.status_code == 200
    assert result.json()["recommendations"]
    assert (
        client.post(
            "/recommend", json={"user_id": "user_1", "timestamp": "2020-01-01T00:00:00Z"}
        ).status_code
        == 422
    )


def test_rate_limit(trained_service):
    with TestClient(create_app(service=trained_service, rate_limit=1)) as client:
        assert client.post("/predict", json=PAYLOAD).status_code == 200
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) >= 1


def test_unavailable_model_returns_503(tmp_path):
    with TestClient(create_app(model_dir=tmp_path)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=PAYLOAD).status_code == 503


def test_redis_cache_and_rate_limit(trained_service):
    cache = CacheStore(client=FakeRedis(decode_responses=True))
    with TestClient(create_app(service=trained_service, cache=cache, rate_limit=2)) as client:
        assert client.get("/health").json()["cache_backend"] == "redis"
        assert client.post("/predict", json=PAYLOAD).status_code == 200
        assert client.post("/predict", json=PAYLOAD).json()["cached"] is True
        assert client.post("/predict", json=PAYLOAD).status_code == 429


def test_redis_outage_local_fallback():
    async def run():
        redis = AsyncMock()
        redis.get.side_effect = ConnectionError("offline")
        redis.setex.side_effect = ConnectionError("offline")
        cache = CacheStore(client=redis, max_entries=2)
        await cache.set("a", {"value": 1})
        assert await cache.get("a") == {"value": 1}
        assert cache.backend == "memory"
        await cache.set("b", 2)
        await cache.set("c", 3)
        assert await cache.get("a") is None
        await cache.set("expired", 4, ttl=-1)
        assert await cache.get("expired") is None

    asyncio.run(run())
