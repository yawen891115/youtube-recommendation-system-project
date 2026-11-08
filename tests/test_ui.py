from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

CLIENT = httpx.Client


def response(request):
    data = {
        "/health": {"model_loaded": True, "cache_backend": "memory"},
        "/monitoring": {"window_requests": 0, "mean_latency_ms": 0, "error_rate": 0, "events": []},
        "/model/info": {
            "feature_importance": [{"feature": "watch_time", "importance": 1}],
            "model_name": "test",
            "model_version": "v1",
            "training_cutoff": "2024-01-01",
        },
        "/predict": {
            "probability": 0.2,
            "confidence": "medium",
            "response_time_ms": 2,
            "cached": False,
            "cold_start": False,
        },
    }
    return httpx.Response(200, json=data[request.url.path])


def test_english_ui_renders_and_predicts():
    def client(**kwargs):
        return CLIENT(transport=httpx.MockTransport(response), **kwargs)

    with patch("httpx.Client", client):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py").run(
            timeout=30
        )
        assert not app.exception
        assert len(app.tabs) == 4
        assert app.tabs[0].label == "Single prediction"
        app.button[0].click().run(timeout=30)
        assert not app.exception
        assert any(metric.label == "Confidence heuristic" for metric in app.metric)


def test_unavailable_api_is_visible():
    def offline(request):
        return httpx.Response(503, json={"detail": "Model unavailable"})

    def client(**kwargs):
        return CLIENT(transport=httpx.MockTransport(offline), **kwargs)

    with patch("httpx.Client", client):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py").run(
            timeout=30
        )
        assert not app.exception
        assert any("503" in error.value for error in app.error)
