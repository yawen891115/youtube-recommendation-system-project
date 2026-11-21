"""Load verified local artifacts and reuse the training feature pipeline online.

Recommendation is two-stage: a cheap retrieval step picks up to ``candidate_limit`` eligible
unseen videos (the user's most-played categories first, then popularity), then the LightGBM
LambdaRank model orders them with the same point-in-time features used offline.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .ranking import score_model

CANDIDATE_LIMIT = 300
TOP_CATEGORIES = 3


class RecommendationService:
    def __init__(self, model_dir="models", candidate_limit=CANDIDATE_LIMIT):
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
        self.candidate_limit = int(candidate_limit)
        self.ranking_columns = self.metadata["ranking_feature_columns"]
        # Online batches are small: one thread avoids thread start-up cost per request.
        for model in (self.model, self.ranker):
            if hasattr(model, "set_params") and "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)
        engineer = self.engineer
        self.video_ids = engineer.video_ids.to_numpy()
        self.log_popularity = np.log1p(engineer.video_history_counts())
        self.seen_codes = {
            user: engineer.video_ids.get_indexer(list(videos))
            for user, videos in engineer.seen.items()
        }

    def predict(self, records):
        frame = self.engineer.transform(pd.DataFrame(records))
        probabilities = self.model.predict_proba(frame[self.metadata["feature_columns"]])[:, 1]
        return [
            {
                "probability": float(np.clip(probability, 0, 1)),
                "confidence": self.confidence(probability),
                "cold_start": bool(user_cold or video_cold),
            }
            for probability, user_cold, video_cold in zip(
                probabilities, frame.user_cold_start, frame.video_cold_start
            )
        ]

    @staticmethod
    def confidence(probability):
        margin = abs(float(probability) - 0.5)
        return "high" if margin >= 0.4 else "medium" if margin >= 0.2 else "low"

    def history(self, user_id, when_ns):
        """Video codes the user played before ``when_ns`` (oldest first)."""
        engineer = self.engineer
        code = engineer.user_ids.get_indexer([user_id])
        n, start = engineer.user_index.lookup(code, np.array([when_ns]))
        return engineer.user_index.kept["video"][start[0] : start[0] + n[0]]

    def recommend(self, user_id, top_k, timestamp):
        when = pd.Timestamp(timestamp)
        when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
        if when < self.engineer.training_cutoff:
            raise ValueError("Recommendation time must be at or after the model training cutoff")
        engineer = self.engineer
        played = self.history(user_id, when.value)
        eligible = engineer.video_upload <= when.value
        seen = self.seen_codes.get(user_id)
        if seen is not None:
            eligible[seen[seen >= 0]] = False
        pool = np.flatnonzero(eligible)
        user = {"history_events": int(len(played)), "top_categories": []}
        if not len(pool):
            return {"user": user, "candidates": 0, "recommendations": []}
        # Retrieval: the user's most-played categories first, then popularity.
        boost = np.zeros(len(engineer.categories))
        if len(played):
            counts = np.bincount(
                engineer.video_category[played], minlength=len(engineer.categories)
            )
            favourite = np.argsort(-counts, kind="stable")[:TOP_CATEGORIES]
            favourite = favourite[counts[favourite] > 0]
            boost[favourite] = 10.0
            user["top_categories"] = [engineer.categories[code] for code in favourite]
        retrieval = self.log_popularity[pool] + boost[engineer.video_category[pool]]
        keep = min(self.candidate_limit, len(pool))
        top = np.argpartition(-retrieval, keep - 1)[:keep]
        candidates = pool[top[np.argsort(-retrieval[top], kind="stable")]]
        requests = pd.DataFrame(
            {
                "user_id": user_id,
                "video_id": self.video_ids[candidates],
                "watch_time": 0.0,
                "timestamp": pd.to_datetime(np.full(len(candidates), when.value), utc=True),
            }
        )
        features = engineer.transform(requests)[self.ranking_columns]
        scores = score_model(self.ranker, features)
        best = np.argsort(-scores, kind="stable")[:top_k]
        chosen = candidates[best]
        return {
            "user": user,
            "candidates": int(len(candidates)),
            "recommendations": [
                {
                    "video_id": str(self.video_ids[code]),
                    "category": engineer.categories[engineer.video_category[code]],
                    "creator_id": int(engineer.videos.creator_id.iloc[code]),
                    "duration_sec": float(engineer.video_duration[code]),
                    "score": float(scores[index]),
                }
                for index, code in zip(best, chosen)
            ],
        }
