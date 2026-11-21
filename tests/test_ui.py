import json
from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

CLIENT = httpx.Client
APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def recommendation(request):
    body = json.loads(request.content)
    new_user = body["user_id"] == "user_999999"
    rows = [
        {
            "video_id": f"video_{n}",
            "category": "tag_1" if n % 2 == 0 else "tag_2",
            "creator_id": n,
            "score": 1.0 - n / 10,
            "duration_sec": 30.0,
        }
        for n in range(body.get("top_k", 10))
    ]
    return {
        "user_id": body["user_id"],
        "cold_start": new_user,
        "history_events": 0 if new_user else 12,
        "top_categories": [] if new_user else ["tag_1"],
        "recommendations": rows,
        "response_time_ms": 3.0,
        "cached": False,
    }


def response(request):
    data = {
        "/health": {"model_loaded": True, "cache_backend": "memory"},
        "/monitoring": {
            "window_requests": 0,
            "mean_latency_ms": 0,
            "p95_latency_ms": 0,
            "error_rate": 0,
            "cache_hit_ratio": 0,
            "events": [],
        },
        "/model/info": {
            "feature_importance": [{"feature": "watch_time", "importance": 1}],
            "model_name": "test",
            "model_version": "v1",
            "training_cutoff": "2024-01-01",
            "ranking_model": {
                "name": "lightgbm_lambdarank",
                "best_baseline": "item_knn",
                "significance_vs_best_baseline": {
                    "mean_difference": 0.02,
                    "low": 0.01,
                    "high": 0.03,
                    "p_value": 0.0001,
                },
            },
            "ranking_benchmark": [
                {
                    "model": "random",
                    "family": "Reference",
                    "test_ndcg@10": 0.05,
                    "test_ndcg@10_low": 0.04,
                    "test_ndcg@10_high": 0.06,
                },
                {
                    "model": "lightgbm_lambdarank",
                    "family": "Tree ensemble (learning to rank)",
                    "test_ndcg@10": 0.11,
                    "test_ndcg@10_low": 0.10,
                    "test_ndcg@10_high": 0.12,
                },
            ],
            "ranking_feature_importance": [
                {"feature": "cowatch_score", "gain": 10.0},
                {"feature": "video_emb_1", "gain": 2.0},
            ],
            "ranking_feature_group_importance": [
                {"group": "Co-watch history", "share": 0.8},
                {"group": "SVD embeddings", "share": 0.2},
            ],
            "ranking_ablations": {
                "without_svd_embeddings": {
                    "removed_features": 101,
                    "test_ndcg@10": 0.11,
                    "difference_vs_final": {"mean_difference": 0.01, "p_value": 0.02},
                }
            },
        },
        "/predict": {
            "probability": 0.2,
            "confidence": "medium",
            "response_time_ms": 2,
            "cached": False,
            "cold_start": False,
        },
    }
    if request.url.path == "/recommend":
        return httpx.Response(200, json=recommendation(request))
    return httpx.Response(200, json=data[request.url.path])


def client(**kwargs):
    return CLIENT(transport=httpx.MockTransport(response), **kwargs)


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_english_ui_renders_and_predicts():
    with patch("httpx.Client", client):
        app = AppTest.from_file(APP).run(timeout=30)
        assert not app.exception
        assert [tab.label for tab in app.tabs] == [
            "Single prediction",
            "Batch prediction",
            "Recommendations",
            "Model insights",
            "Monitoring",
        ]
        button(app, "Predict like probability").click().run(timeout=30)
        assert not app.exception
        assert any(metric.label == "Confidence heuristic" for metric in app.metric)


def test_recommendations_show_cold_start_and_history():
    with patch("httpx.Client", client):
        app = AppTest.from_file(APP).run(timeout=30)
        app.text_input[2].set_value("user_999999")
        button(app, "Recommend videos").click().run(timeout=30)
        assert not app.exception
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["Cold start"] == "True" and metrics["Earlier plays"] == "0"
        assert any("No watch history" in info.value for info in app.info)


def test_model_insights_shows_benchmark_and_importance():
    with patch("httpx.Client", client):
        app = AppTest.from_file(APP).run(timeout=30)
        assert not app.exception
        tables = [frame.value for frame in app.dataframe]
        assert any("test_ndcg@10" in table for table in tables)
        assert any("Paired bootstrap" in caption.value for caption in app.caption)
        assert any("final_minus_variant" in table for table in tables)


def test_unavailable_api_is_visible():
    def offline(request):
        return httpx.Response(503, json={"detail": "Model unavailable"})

    def offline_client(**kwargs):
        return CLIENT(transport=httpx.MockTransport(offline), **kwargs)

    with patch("httpx.Client", offline_client):
        app = AppTest.from_file(APP).run(timeout=30)
        assert not app.exception
        assert any("503" in error.value for error in app.error)
