"""Exercise the running API, Streamlit, MLflow and Prometheus services."""

import argparse
import json
import urllib.request


def get(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        return (
            json.loads(body)
            if "application/json" in response.headers.get("Content-Type", "")
            else body.decode()
        )


def run(api="http://127.0.0.1:8000", stack=False):
    health = get(api + "/health")
    assert health["model_loaded"], health
    # user_16 is in the bundled KuaiRand subset; requests replay a time after its last play.
    payload = {
        "user_id": "user_16",
        "video_id": "video_1",
        "watch_time": 20,
        "timestamp": "2022-05-08T16:00:00Z",
    }
    prediction = get(api + "/predict", payload)
    assert 0 <= prediction["probability"] <= 1
    repeat = get(api + "/predict", payload)
    assert repeat["cached"]
    batch = get(api + "/predict/batch", {"instances": [payload, payload]})
    assert batch["count"] == 2
    assert all(row["probability"] == prediction["probability"] for row in batch["predictions"])
    request = {"user_id": "user_16", "top_k": 5, "timestamp": "2022-05-08T16:00:00Z"}
    recommendations = get(api + "/recommend", request)
    assert len(recommendations["recommendations"]) == 5
    assert not recommendations["cold_start"] and recommendations["top_categories"]
    assert get(api + "/recommend", request)["cached"]
    cold = get(api + "/recommend", request | {"user_id": "user_9999999"})
    assert cold["cold_start"] and cold["history_events"] == 0
    metrics = get(api + "/metrics")
    for name in [
        "recommendation_predictions_total",
        "recommendation_http_duration_seconds_bucket",
        "recommendation_cache_requests_total",
        "recommendation_model_seconds_bucket",
    ]:
        assert name in metrics, name
    assert get(api + "/monitoring")["window_requests"] >= 4
    info = get(api + "/model/info")
    assert "model_name" in info and info["ranking_model"]["name"] == "lightgbm_lambdarank"
    if stack:
        assert health["cache_backend"] == "redis"
        assert "ok" in get("http://127.0.0.1:8501/_stcore/health").lower()
        assert "<html" in get("http://127.0.0.1:8501").lower()
        assert "ok" in get("http://127.0.0.1:5000/health").lower()
        assert "healthy" in get("http://127.0.0.1:9090/-/healthy").lower()
        rules = get("http://127.0.0.1:9090/api/v1/rules")["data"]["groups"]
        assert any(group["name"] == "recommendation-api" for group in rules)
    print(
        json.dumps(
            {
                "status": "passed",
                "model_version": prediction["model_version"],
                "cache_backend": health["cache_backend"],
                "stack_checked": stack,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--stack", action="store_true")
    run(**vars(parser.parse_args()))
