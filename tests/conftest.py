import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import OBSERVED_FEATURES, FeatureEngineer
from src.inference import RecommendationService


@pytest.fixture(scope="session")
def sample_data():
    users = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2", "user_3"],
            "age": [24, 31, 42],
            "gender": ["F", "M", "Other"],
            "preferred_category": ["Music", "Gaming", "Comedy"],
            "join_date": pd.to_datetime(["2023-01-01"] * 3, utc=True),
        }
    )
    videos = pd.DataFrame(
        {
            "video_id": [f"video_{i}" for i in range(1, 7)],
            "creator_id": list(range(6)),
            "category": ["Music", "Gaming", "Comedy", "Food", "Sports", "Travel"],
            "duration_sec": [30.0] * 6,
            "view_count": [100] * 6,
            "upload_date": pd.to_datetime(["2023-01-01"] * 6, utc=True),
        }
    )
    events = pd.DataFrame(
        {
            "user_id": [f"user_{i % 3 + 1}" for i in range(100)],
            "video_id": [f"video_{i % 4 + 1}" for i in range(100)],
            "watch_time": (np.arange(100) % 30 + 1).astype(float),
            "liked": np.arange(100) % 2,
            "shared": (np.arange(100) % 7 == 0).astype(int),
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
    ranking = [key for key in features if key not in OBSERVED_FEATURES]
    model = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42).fit(
        features, events.iloc[:64].liked
    )
    ranker = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42).fit(
        features[ranking], events.iloc[:64].liked
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
        "artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).iterdir()
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return RecommendationService(root)
