import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import FeatureEngineer
from src.inference import RecommendationService
from src.ranking import group_sizes, ranking_columns


@pytest.fixture(scope="session")
def sample_data():
    """Small tables in the project schema (same columns as the KuaiRand-Pure subset)."""
    users = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2", "user_3"],
            "user_active_degree": ["full_active", "high_active", "middle_active"],
            "is_lowactive_period": [0, 0, 1],
            "is_live_streamer": [0, 1, 0],
            "is_video_author": [1, 0, 0],
            "follow_user_num": [10, 200, 0],
            "fans_user_num": [5, 1000, 0],
            "friend_user_num": [1, 30, 0],
            "register_days": [400, 900, 30],
        }
    )
    videos = pd.DataFrame(
        {
            "video_id": [f"video_{i}" for i in range(1, 7)],
            "creator_id": [0, 0, 1, 2, 3, 3],
            "category": [f"tag_{i}" for i in range(1, 7)],
            "duration_sec": [30.0] * 6,
            "video_type": ["NORMAL"] * 6,
            "upload_type": ["Web", "Web", "LongImport", "Web", "Kmovie", "Web"],
            "upload_date": pd.to_datetime(["2023-01-01"] * 6, utc=True),
        }
    )
    watch = (np.arange(100) % 30 + 1).astype(float)
    events = pd.DataFrame(
        {
            "user_id": [f"user_{i % 3 + 1}" for i in range(100)],
            "video_id": [f"video_{i % 4 + 1}" for i in range(100)],
            "watch_time": watch,
            "liked": np.arange(100) % 2,
            "shared": (np.arange(100) % 7 == 0).astype(int),
            "long_view": (watch >= 15).astype(int),
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC"),
        }
    )
    return users, videos, events


@pytest.fixture(scope="session")
def trained_service(tmp_path_factory, sample_data):
    users, videos, events = sample_data
    root = tmp_path_factory.mktemp("models")
    engineer = FeatureEngineer(dimensions=5).fit(events.iloc[:64], users, videos)
    features = engineer.transform(events.iloc[:64])
    ranking = ranking_columns(features.columns)
    model = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42).fit(
        features, events.iloc[:64].liked
    )
    # A real LambdaRank model on per-user query groups, like the production ranker.
    by_user = events.iloc[:64].reset_index(drop=True).sort_values("user_id", kind="stable")
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=10, min_child_samples=2, verbosity=-1
    ).fit(
        features.loc[by_user.index, ranking],
        by_user.liked,
        group=group_sizes(by_user.user_id.astype("category").cat.codes),
    )
    engineer.save_artifacts(root)
    joblib.dump(model, root / "best_model_test.pkl")
    joblib.dump(ranker, root / "ranking_model.pkl")
    metadata = {
        "model_file": "best_model_test.pkl",
        "model_version": "test-v1",
        "model_name": "test_forest",
        "feature_columns": list(features),
        "ranking_feature_columns": ranking,
        "training_cutoff": engineer.training_cutoff.isoformat(),
        "embedding_cutoff": engineer.embedding_cutoff.isoformat(),
        "feature_importance": [
            {"feature": key, "importance": float(value)}
            for key, value in zip(features, model.feature_importances_)
        ],
        "validation_metrics": {},
        "test_metrics": {},
        "ranking_model": {
            "name": "lightgbm_lambdarank",
            "test_metrics": {"ndcg@10": 0.1},
            "best_baseline": "item_knn",
        },
        "ranking_benchmark": [
            {"model": "random", "family": "Reference", "test_ndcg@10": 0.05},
            {"model": "lightgbm_lambdarank", "family": "Tree ensemble", "test_ndcg@10": 0.1},
        ],
        "ranking_feature_importance": [
            {"feature": key, "gain": float(value)}
            for key, value in zip(ranking, ranker.booster_.feature_importance("gain"))
        ],
        "ranking_feature_group_importance": [{"group": "User watch history", "share": 1}],
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).iterdir()
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return RecommendationService(root)
