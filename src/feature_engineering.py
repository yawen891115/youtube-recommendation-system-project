"""Shared point-in-time features for training and online inference."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .data_validation import CATEGORIES

SECONDS_PER_DAY = 86400
OBSERVED_FEATURES = [
    "watch_time",
    "completion_rate",
    "watch_time_normalized",
    "popularity_watch_ratio",
]


def timestamps(values):
    return pd.to_datetime(values, utc=True, errors="raise").astype("int64").to_numpy()


class FeatureEngineer:
    def __init__(self, dimensions=50, seed=42, warmup_fraction=0.10):
        self.dimensions, self.seed, self.warmup_fraction = dimensions, seed, warmup_fraction
        self.scalers, self.encoders, self.embeddings = {}, {}, {}

    def _history(self, frame, key, sessions=False):
        histories = {}
        for value, group in frame.groupby(key, sort=False):
            group = group.sort_values("timestamp", kind="stable")
            times = timestamps(group.timestamp)
            watch = group.watch_time.to_numpy(float)
            history = {
                "times": times,
                "watch": np.r_[0.0, watch.cumsum()],
                "square": np.r_[0.0, (watch**2).cumsum()],
            }
            for column in ["liked", "shared"]:
                history[column] = np.r_[0.0, group[column].to_numpy(float).cumsum()]
            if sessions:
                new = np.r_[True, np.diff(times) > 30 * 60 * 1e9]
                starts = np.maximum.accumulate(np.where(new, np.arange(len(times)), 0))
                history.update(session_start=starts, session_id=np.cumsum(new) - 1)
                categories = group.video_id.map(self.videos.category).fillna("unknown")
                category_matrix = np.stack(
                    [(categories == c).to_numpy(int) for c in CATEGORIES], axis=1
                )
                history["category"] = np.vstack(
                    [np.zeros((1, len(CATEGORIES))), category_matrix.cumsum(axis=0)]
                )
            histories[value] = history
        return histories

    def fit(self, interactions, users, videos):
        if not 0 < self.warmup_fraction < 1 or self.dimensions < 1:
            raise ValueError("Invalid embedding settings")
        train = interactions.sort_values("timestamp", kind="stable").copy()
        self.users = users.set_index("user_id").copy()
        self.videos = videos.set_index("video_id").copy()
        self.training_cutoff = pd.to_datetime(train.timestamp, utc=True).max()
        self.user_history = self._history(train, "user_id", True)
        self.video_history = self._history(train, "video_id")
        self.seen = train.groupby("user_id").video_id.agg(set).to_dict()
        self.scalers["watch_time"] = MinMaxScaler().fit(train[["watch_time"]])
        self.default_age = float(users.age.median())
        self.default_duration = float(videos.duration_sec.median())
        self.encoders["categorical"] = OneHotEncoder(
            categories=[
                ["M", "F", "Other", "unknown"],
                CATEGORIES + ["unknown"],
                CATEGORIES + ["unknown"],
                ["night", "morning", "afternoon", "evening"],
            ],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self.encoders["categorical"].fit(
            pd.DataFrame(
                [["M", "Music", "Music", "night"]],
                columns=["gender", "category", "preferred_category", "day_period"],
            )
        )
        self._fit_embeddings(train)
        self.feature_columns = self.transform(train.iloc[:1]).columns.tolist()
        return self

    def _fit_embeddings(self, train):
        # A fixed early-history basis avoids current-label and future-event leakage.
        cutoff_index = min(len(train) - 1, max(0, int(len(train) * self.warmup_fraction) - 1))
        self.embedding_cutoff = pd.to_datetime(train.iloc[cutoff_index].timestamp, utc=True)
        warm = train[pd.to_datetime(train.timestamp, utc=True) <= self.embedding_cutoff]
        user_ids, video_ids = list(self.users.index), list(self.videos.index)
        user_map = {value: n for n, value in enumerate(user_ids)}
        video_map = {value: n for n, value in enumerate(video_ids)}
        matrix = csr_matrix(
            (
                warm.liked.to_numpy(np.float32),
                (warm.user_id.map(user_map), warm.video_id.map(video_map)),
            ),
            shape=(len(user_ids), len(video_ids)),
        )
        components = min(self.dimensions, min(matrix.shape) - 1)
        if components > 0 and matrix.nnz and matrix.data.sum() > 0:
            svd = TruncatedSVD(n_components=components, random_state=self.seed)
            u, v = svd.fit_transform(matrix), None
            v = svd.components_.T
        else:
            svd, components = None, 0
            u, v = np.zeros((len(user_ids), 0)), np.zeros((len(video_ids), 0))
        u = np.pad(u, ((0, 0), (0, self.dimensions - components))).astype(np.float32)
        v = np.pad(v, ((0, 0), (0, self.dimensions - components))).astype(np.float32)
        self.embeddings = {
            "user_vectors": u,
            "video_vectors": v,
            "user_map": user_map,
            "video_map": video_map,
            "svd_model": svd,
        }

    def _aggregate(self, frame, key, histories, prefix, sessions=False):
        columns = [
            "watch_mean",
            "watch_std",
            "watch_count",
            "liked_mean",
            "liked_sum",
            "shared_mean",
            "shared_sum",
            "recency_days",
            "tenure_days",
        ]
        if sessions:
            columns += [
                "session_id",
                "session_length",
                "session_duration_minutes",
                "recent_engagement",
                "category_diversity",
            ]
        result = pd.DataFrame(0.0, index=frame.index, columns=[prefix + c for c in columns])
        if sessions:
            result[prefix + "session_length"] = 1.0
        for value, rows in frame.groupby(key, sort=False):
            history = histories.get(value)
            if history is None:
                continue
            query = timestamps(rows.timestamp)
            n = np.searchsorted(history["times"], query, side="left")
            valid = n > 0
            safe_n = np.maximum(n, 1)
            sums, squares = history["watch"][n], history["square"][n]
            out = {
                "watch_mean": sums / safe_n,
                "watch_std": np.sqrt(
                    np.maximum(0, (squares - sums**2 / safe_n) / np.maximum(n - 1, 1))
                ),
                "watch_count": n,
                "liked_sum": history["liked"][n],
                "shared_sum": history["shared"][n],
                "liked_mean": history["liked"][n] / safe_n,
                "shared_mean": history["shared"][n] / safe_n,
                "recency_days": np.where(
                    valid,
                    (query - history["times"][np.maximum(n - 1, 0)]) / 1e9 / SECONDS_PER_DAY,
                    0,
                ),
                "tenure_days": np.where(
                    valid, (query - history["times"][0]) / 1e9 / SECONDS_PER_DAY, 0
                ),
            }
            if sessions:
                last = np.maximum(n - 1, 0)
                same = valid & ((query - history["times"][last]) <= 30 * 60 * 1e9)
                start = history["session_start"][last]
                out.update(
                    session_id=np.where(valid, history["session_id"][last] + (~same), 0),
                    session_length=np.where(same, n - start + 1, 1),
                    session_duration_minutes=np.where(
                        same, (query - history["times"][start]) / 1e9 / 60, 0
                    ),
                    recent_engagement=(
                        (history["liked"][n] - history["liked"][np.maximum(n - 5, 0)])
                        + (history["shared"][n] - history["shared"][np.maximum(n - 5, 0)])
                    )
                    / np.maximum(np.minimum(n, 5), 1),
                    category_diversity=(history["category"][n] > 0).sum(axis=1),
                )
            result.loc[rows.index, [prefix + column for column in out]] = np.column_stack(
                list(out.values())
            )
        return result

    def transform(self, requests):
        frame = (
            requests[["user_id", "video_id", "watch_time", "timestamp"]]
            .copy()
            .reset_index(drop=True)
        )
        frame.timestamp = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        if not np.isfinite(frame.watch_time).all() or (frame.watch_time < 0).any():
            raise ValueError("watch_time must be finite and nonnegative")
        users = self.users.reindex(frame.user_id).reset_index(drop=True)
        videos = self.videos.reindex(frame.video_id).reset_index(drop=True)
        hour = frame.timestamp.dt.hour
        day = frame.timestamp.dt.dayofweek
        duration = videos.duration_sec.fillna(self.default_duration).clip(lower=1e-6)
        result = pd.DataFrame(
            {
                "watch_time": frame.watch_time,
                "age": users.age.fillna(self.default_age),
                "duration_sec": duration,
                "hour_of_day": hour,
                "day_of_week": day,
                "is_weekend": (day >= 5).astype(int),
                "month": frame.timestamp.dt.month,
                "completion_rate": (frame.watch_time / duration).clip(0, 1),
                "watch_time_normalized": self.scalers["watch_time"]
                .transform(frame[["watch_time"]])
                .ravel(),
                "hour_weekend_interaction": hour * (day >= 5),
                "user_cold_start": (~frame.user_id.isin(self.user_history)).astype(int),
                "video_cold_start": (~frame.video_id.isin(self.video_history)).astype(int),
            }
        )
        upload = pd.to_datetime(videos.upload_date, utc=True)
        result["video_age_days"] = (
            ((frame.timestamp - upload).dt.total_seconds() / SECONDS_PER_DAY)
            .fillna(0)
            .clip(lower=0)
        )
        for values in [
            self._aggregate(frame, "user_id", self.user_history, "user_", True),
            self._aggregate(frame, "video_id", self.video_history, "video_"),
        ]:
            result = pd.concat([result, values], axis=1)
        result["user_cold_start"] = (result.user_watch_count == 0).astype(int)
        result["video_cold_start"] = (result.video_watch_count == 0).astype(int)
        result["recency_days"] = result.user_recency_days
        result["interactions_per_day"] = result.user_watch_count / (result.user_tenure_days + 1)
        result["video_popularity"] = result.video_watch_count
        result["engagement_rate"] = (
            result.video_liked_sum + result.video_shared_sum
        ) / result.video_watch_count.clip(lower=1)
        result["popularity_watch_ratio"] = result.video_popularity / (frame.watch_time + 1)
        categorical = pd.DataFrame(
            {
                "gender": users.gender.fillna("unknown"),
                "category": videos.category.fillna("unknown"),
                "preferred_category": users.preferred_category.fillna("unknown"),
                "day_period": np.select(
                    [hour < 6, hour < 12, hour < 18],
                    ["night", "morning", "afternoon"],
                    default="evening",
                ),
            }
        )
        encoder = self.encoders["categorical"]
        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    encoder.transform(categorical),
                    columns=encoder.get_feature_names_out(),
                    index=result.index,
                ),
            ],
            axis=1,
        )
        available = (frame.timestamp > self.embedding_cutoff).to_numpy()
        for kind, column in [("user", "user_id"), ("video", "video_id")]:
            indices = (
                frame[column].map(self.embeddings[kind + "_map"]).fillna(-1).astype(int).to_numpy()
            )
            vectors = self.embeddings[kind + "_vectors"]
            embedded = vectors[np.maximum(indices, 0)].copy()
            embedded[(indices < 0) | ~available] = 0
            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        embedded, columns=[f"{kind}_emb_{n}" for n in range(self.dimensions)]
                    ),
                ],
                axis=1,
            )
        return result.astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0)

    def save_artifacts(self, directory="models"):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, value in [
            ("scalers.pkl", self.scalers),
            ("encoders.pkl", self.encoders),
            ("embeddings.pkl", self.embeddings),
            ("feature_columns.pkl", self.feature_columns),
            ("feature_engineer.joblib", self),
        ]:
            joblib.dump(value, root / name, compress=3)
