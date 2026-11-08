"""Load verified local artifacts and reuse the training feature pipeline."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class RecommendationService:
    def __init__(self, model_dir="models"):
        root = Path(model_dir)
        self.metadata = json.loads((root / "metadata.json").read_text())
        files = [self.metadata["model_file"], "ranking_model.pkl", "feature_engineer.joblib"]
        for name in files:
            if Path(name).name != name:
                raise ValueError("Artifact names must be local file names")
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != self.metadata["artifact_sha256"].get(name):
                raise ValueError(f"Artifact checksum mismatch: {name}")
        # Joblib artifacts must come from this trusted repository or your own training run.
        self.model = joblib.load(root / self.metadata["model_file"])
        self.ranker = joblib.load(root / "ranking_model.pkl")
        self.engineer = joblib.load(root / "feature_engineer.joblib")
        self.version = self.metadata["model_version"]
        self.candidates = self.engineer.videos.copy()
        self.candidates["history_count"] = [
            len(self.engineer.video_history.get(key, {}).get("times", []))
            for key in self.candidates.index
        ]

    def predict(self, records):
        frame = self.engineer.transform(pd.DataFrame(records))
        with threadpool_limits(limits=2):
            probabilities = self.model.predict_proba(frame[self.metadata["feature_columns"]])[:, 1]
        return [
            {
                "probability": float(np.clip(probability, 0, 1)),
                "confidence": self.confidence(probability),
                "cold_start": bool(row.user_cold_start or row.video_cold_start),
            }
            for probability, row in zip(probabilities, frame.itertuples())
        ]

    @staticmethod
    def confidence(probability):
        margin = abs(float(probability) - 0.5)
        return "high" if margin >= 0.4 else "medium" if margin >= 0.2 else "low"

    def recommend(self, user_id, top_k, timestamp):
        when = pd.Timestamp(timestamp)
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        if when < self.engineer.training_cutoff:
            raise ValueError("Recommendation time must be at or after the model training cutoff")
        catalog = self.candidates.copy()
        catalog = catalog[pd.to_datetime(catalog.upload_date, utc=True) <= when]
        catalog = catalog[~catalog.index.isin(self.engineer.seen.get(user_id, set()))]
        preferred = (
            self.engineer.users.loc[user_id, "preferred_category"]
            if user_id in self.engineer.users.index
            else None
        )
        catalog["preferred"] = (catalog.category == preferred).astype(int)
        catalog = catalog.sort_values(
            ["preferred", "history_count"], ascending=False, kind="stable"
        ).head(300)
        if catalog.empty:
            return []
        requests = pd.DataFrame(
            {"user_id": user_id, "video_id": catalog.index, "watch_time": 0.0, "timestamp": when}
        )
        features = self.engineer.transform(requests)[self.metadata["ranking_feature_columns"]]
        with threadpool_limits(limits=2):
            scores = self.ranker.predict_proba(features)[:, 1]
        catalog["score"] = scores
        return [
            {
                "video_id": video_id,
                "category": row.category,
                "duration_sec": float(row.duration_sec),
                "score": float(row.score),
            }
            for video_id, row in catalog.sort_values("score", ascending=False, kind="stable")
            .head(top_k)
            .iterrows()
        ]
