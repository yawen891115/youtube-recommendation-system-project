"""Shared point-in-time features for training, offline ranking evaluation and online serving.

Every history feature for a request at time ``t`` uses only plays strictly before ``t``: the
current play and every later play are excluded. Learned inputs (priors, scaler, SVD basis,
co-watch similarity) come from the training plays only. The play log that histories read can
extend past the training cutoff, as an online feature store would: offline evaluation and
serving use the full log, and a request after its end sees the log's final state.

History lookups are vectorised: plays are stored once, sorted by (entity, time), with running
sums. One ``searchsorted`` call finds how many plays each request may see, so scoring 300
candidates online or building 2 million offline candidate rows uses the same code path.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .data_validation import ACTIVITY_LEVELS

SECONDS_PER_DAY = 86400
NS_PER_DAY = SECONDS_PER_DAY * 10**9
NS_PER_HOUR = 3600 * 10**9
NS_PER_MINUTE = 60 * 10**9
SESSION_GAP_NS = 30 * NS_PER_MINUTE

# Known only after the video has been watched; excluded from the pre-watch ranker.
OBSERVED_FEATURES = [
    "watch_time",
    "completion_rate",
    "watch_time_normalized",
    "popularity_watch_ratio",
]

# Watch-history features beyond plain per-user / per-video aggregates (all pre-watch).
WATCH_HISTORY_FEATURES = [
    "user_cat_watch_count",
    "user_cat_like_rate",
    "user_cat_completion",
    "user_cat_decayed_share",
    "user_cat_decayed_completion",
    "user_creator_watch_count",
    "user_creator_decayed_share",
    "user_completion_mean",
    "user_long_view_rate",
    "video_completion_mean",
    "video_long_view_rate",
    "video_plays_24h",
    "cowatch_score",
    "cowatch_mean",
]
USER_NUMERIC = [
    "register_days",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "is_video_author",
    "is_live_streamer",
    "is_lowactive_period",
]
COUNT_COLUMNS = {"follow_user_num", "fans_user_num", "friend_user_num"}


def timestamps(values):
    """Convert timestamps to int64 nanoseconds since the epoch (UTC)."""
    values = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        converted = values.dt.tz_convert("UTC")
    else:
        converted = pd.to_datetime(values, utc=True, errors="raise")
    return converted.dt.as_unit("ns").astype("int64").to_numpy()


class HistoryIndex:
    """Time-sorted plays of one entity type with vectorised "plays before t" lookups.

    Plays are sorted by (entity code, time). Entity ``e`` owns rows
    ``[offsets[e], offsets[e] + counts[e])``. Each play time is replaced by its rank among the
    distinct play times, so ``code * stride + rank`` is a sorted composite key and one
    ``searchsorted`` answers every query exactly (strictly-before semantics, ties excluded).
    Running sums restart for every entity, which keeps them as precise as per-entity loops.
    """

    def __init__(self, codes, times, n_entities, values, sessions=False, keep=None):
        codes = np.asarray(codes, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        if len(codes) and (codes.min() < 0 or codes.max() >= n_entities):
            raise ValueError("History contains entities outside the known catalog")
        order = np.lexsort((times, codes))
        codes, times = codes[order], times[order]
        self.n_entities = int(n_entities)
        self.times = times
        self.counts = np.bincount(codes, minlength=n_entities).astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)
        self.unique_times = np.unique(times)
        self.stride = np.int64(len(self.unique_times) + 1)
        self.composite = codes * self.stride + np.searchsorted(self.unique_times, times)
        self.running = {
            name: pd.Series(np.asarray(value, dtype=np.float64)[order])
            .groupby(codes)
            .cumsum()
            .to_numpy()
            for name, value in values.items()
        }
        self.kept = {name: np.asarray(value)[order] for name, value in (keep or {}).items()}
        if sessions:
            position = np.arange(len(codes)) - self.offsets[codes]
            gap = np.concatenate([[True], np.diff(times) > SESSION_GAP_NS])
            new = (position == 0) | gap
            self.session_start = np.maximum.accumulate(np.where(new, np.arange(len(codes)), 0))
            counter = np.cumsum(new)
            self.session_id = counter - counter[self.offsets[codes]]

    def lookup(self, codes, query_times):
        """Return (n, start): visible play count and the entity's first global row."""
        codes = np.asarray(codes, dtype=np.int64)
        known = codes >= 0
        safe = np.where(known, codes, 0)
        ranks = np.searchsorted(self.unique_times, query_times, side="left")
        position = np.searchsorted(self.composite, safe * self.stride + ranks, side="left")
        start = self.offsets[safe]
        return np.where(known, position - start, 0), start

    def total(self, name, start, n):
        """Sum of ``name`` over each entity's first ``n`` plays."""
        running = self.running[name]
        if not len(running):
            return np.zeros(len(n))
        return np.where(n > 0, running[np.clip(start + n - 1, 0, len(running) - 1)], 0.0)

    def time_at(self, index):
        if not len(self.times):
            return np.zeros(len(index), dtype=np.int64)
        return self.times[np.clip(index, 0, len(self.times) - 1)]


def first_occurrences(codes, times, keys):
    """1.0 for the first play of each (entity, key) pair in time order, else 0.0."""
    frame = pd.DataFrame({"code": codes, "time": times, "key": keys})
    ordered = frame.sort_values(["code", "time"], kind="stable")
    flags = np.zeros(len(frame))
    flags[ordered.index.to_numpy()] = (~ordered.duplicated(["code", "key"])).to_numpy() & (
        ordered.key.to_numpy() >= 0
    )
    return flags


class FeatureEngineer:
    def __init__(
        self,
        dimensions=50,
        seed=42,
        warmup_fraction=0.10,
        half_life_days=30.0,
        smoothing=2.0,
    ):
        self.dimensions, self.seed, self.warmup_fraction = dimensions, seed, warmup_fraction
        self.half_life_days, self.smoothing = half_life_days, smoothing
        self.scalers, self.encoders, self.embeddings = {}, {}, {}

    # ------------------------------------------------------------------ fitting
    def fit(self, interactions, users, videos, history=None):
        """Fit on the training plays; read histories from ``history`` (default: the same).

        Model inputs learned from data (priors, scaler, SVD, co-watch similarity) use only
        ``interactions``. ``history`` is the play log that point-in-time features may read:
        each request still sees only plays strictly before its own timestamp.
        """
        if not 0 < self.warmup_fraction < 1 or self.dimensions < 1:
            raise ValueError("Invalid embedding settings")
        if self.half_life_days <= 0 or self.smoothing <= 0:
            raise ValueError("Invalid history settings")
        train = interactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
        log = train if history is None else history.sort_values("timestamp", kind="stable")
        log = log.reset_index(drop=True)
        self.users = users.set_index("user_id").copy()
        self.videos = videos.set_index("video_id").copy()
        self.user_ids = pd.Index(self.users.index)
        self.video_ids = pd.Index(self.videos.index)
        self.categories = sorted(self.videos.category.astype(str).unique())
        self.category_index = {name: code for code, name in enumerate(self.categories)}
        self.video_category = self.videos.category.map(self.category_index).to_numpy(np.int64)
        self.creator_ids = pd.Index(self.videos.creator_id.unique())
        self.video_creator = self.creator_ids.get_indexer(self.videos.creator_id).astype(np.int64)
        self.video_duration = self.videos.duration_sec.astype(float).to_numpy()
        self.video_upload = timestamps(self.videos.upload_date)
        self.user_profile = {
            column: self.users[column].astype(float).to_numpy() for column in USER_NUMERIC
        }
        self.profile_defaults = {
            column: float(np.median(values)) for column, values in self.user_profile.items()
        }
        self.user_activity = self.users.user_active_degree.astype(str).to_numpy(object)
        self.training_cutoff = pd.to_datetime(train.timestamp, utc=True).max()
        self.history_end = pd.to_datetime(log.timestamp, utc=True).max()
        self.default_duration = float(videos.duration_sec.median())

        train_users = self.user_ids.get_indexer(train.user_id)
        train_videos = self.video_ids.get_indexer(train.video_id)
        times = timestamps(log.timestamp)
        user_codes = self.user_ids.get_indexer(log.user_id)
        video_codes = self.video_ids.get_indexer(log.video_id)
        if min(train_users.min(), train_videos.min(), user_codes.min(), video_codes.min()) < 0:
            raise ValueError("Interactions reference unknown users or videos")
        watch = log.watch_time.to_numpy(float)
        liked = log.liked.to_numpy(float)
        shared = log.shared.to_numpy(float)
        long_view = log.long_view.to_numpy(float)
        completion = np.clip(watch / np.maximum(self.video_duration[video_codes], 1e-6), 0, 1)
        categories = self.video_category[video_codes]
        creators = self.video_creator[video_codes]
        # Recency weights relative to the end of the history, so every weight is <= 1.
        cutoff_ns = int(times.max())
        self.cutoff_ns = cutoff_ns
        self.decay_scale_ns = self.half_life_days * NS_PER_DAY / np.log(2)
        decay = np.exp((times - cutoff_ns) / self.decay_scale_ns)
        train_completion = np.clip(
            train.watch_time.to_numpy(float) / np.maximum(self.video_duration[train_videos], 1e-6),
            0,
            1,
        )
        self.priors = {
            "like_rate": float(train.liked.mean()),
            "completion": float(train_completion.mean()),
            "long_view": float(train.long_view.mean()),
        }

        self.user_index = HistoryIndex(
            user_codes,
            times,
            len(self.user_ids),
            {
                "watch": watch,
                "square": watch**2,
                "liked": liked,
                "shared": shared,
                "completion": completion,
                "long_view": long_view,
                "decay": decay,
                "new_category": first_occurrences(user_codes, times, categories),
            },
            sessions=True,
            keep={"video": video_codes},
        )
        self.video_index = HistoryIndex(
            video_codes,
            times,
            len(self.video_ids),
            {
                "watch": watch,
                "square": watch**2,
                "liked": liked,
                "shared": shared,
                "completion": completion,
                "long_view": long_view,
            },
        )
        # One entity per (user, category): time-aware category watch histories.
        n_categories = len(self.categories)
        self.user_category_index = HistoryIndex(
            user_codes * n_categories + categories,
            times,
            len(self.user_ids) * n_categories,
            {
                "liked": liked,
                "completion": completion,
                "decay": decay,
                "decay_completion": decay * completion,
            },
        )
        # One entity per (user, creator) pair seen in training (compact codes).
        pair_keys = user_codes * len(self.creator_ids) + creators
        self.user_creator_pairs = pd.Index(np.unique(pair_keys))
        self.user_creator_index = HistoryIndex(
            self.user_creator_pairs.get_indexer(pair_keys),
            times,
            len(self.user_creator_pairs),
            {"decay": decay},
        )
        self._fit_cowatch(train_users, train_videos)
        self.seen = log.groupby("user_id").video_id.agg(set).to_dict()
        self.scalers["watch_time"] = MinMaxScaler().fit(train[["watch_time"]])
        self.encoders["categorical"] = OneHotEncoder(
            categories=[
                ACTIVITY_LEVELS,
                self.categories + ["unknown"],
                sorted(self.videos.upload_type.astype(str).unique()) + ["unknown"],
                sorted(self.videos.video_type.astype(str).unique()) + ["unknown"],
                ["night", "morning", "afternoon", "evening"],
            ],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self.encoders["categorical"].fit(
            pd.DataFrame(
                [
                    [
                        "UNKNOWN",
                        "unknown",
                        "unknown",
                        "unknown",
                        "night",
                    ]
                ],
                columns=[
                    "user_active_degree",
                    "category",
                    "upload_type",
                    "video_type",
                    "day_period",
                ],
            )
        )
        self._fit_embeddings(
            train,
            train_users,
            train_videos,
            train.liked.to_numpy(float),
            train.shared.to_numpy(float),
        )
        self.feature_columns = self.transform(train.iloc[:1]).columns.tolist()
        return self

    def _fit_cowatch(self, user_codes, video_codes):
        """Item-item co-watch counts from the training plays (distinct user-video pairs)."""
        n_videos = len(self.video_ids)
        keys = np.unique(user_codes * n_videos + video_codes)
        matrix = csr_matrix(
            (np.ones(len(keys), dtype=np.float32), (keys // n_videos, keys % n_videos)),
            shape=(len(self.user_ids), n_videos),
        )
        self.cowatch = (matrix.T @ matrix).toarray().astype(np.float32)
        self.cowatch_counts = np.diag(self.cowatch).copy()
        self.train_pairs = keys

    def _played_in_training(self, keys):
        """Membership in the sorted training (user, video) keys by binary search."""
        position = np.searchsorted(self.train_pairs, keys)
        found = np.minimum(position, len(self.train_pairs) - 1)
        return (position < len(self.train_pairs)) & (self.train_pairs[found] == keys)

    def cowatch_scores(self, user_codes, video_codes, query, leave_user_out=True):
        """Sum and mean cosine similarity between each candidate and the user's earlier plays.

        With ``leave_user_out`` the user's own training plays are removed from the counts, so
        a training positive never scores its own co-occurrence (no target leakage).
        """
        n, start = self.user_index.lookup(user_codes, query)
        rows = np.flatnonzero((n > 0) & (video_codes >= 0))
        total = np.zeros(len(query))
        if len(rows):
            lengths = n[rows]
            owner = np.repeat(rows, lengths)
            offsets = np.repeat(start[rows] - np.cumsum(np.r_[0, lengths[:-1]]), lengths)
            history = self.user_index.kept["video"][np.arange(len(owner)) + offsets]
            candidate = video_codes[owner]
            together = self.cowatch[history, candidate].astype(np.float64)
            norm_history = self.cowatch_counts[history].astype(np.float64)
            norm_candidate = self.cowatch_counts[candidate].astype(np.float64)
            if leave_user_out:
                n_videos = len(self.video_ids)
                users = user_codes[owner]
                own_history = self._played_in_training(users * n_videos + history)
                own_candidate = self._played_in_training(users * n_videos + candidate)
                together -= own_history & own_candidate
                norm_history -= own_history
                norm_candidate -= own_candidate
            denominator = np.sqrt(np.maximum(norm_history, 0) * np.maximum(norm_candidate, 0))
            similarity = np.divide(
                together, denominator, out=np.zeros(len(owner)), where=denominator > 0
            )
            similarity[history == candidate] = 0.0
            total = np.bincount(owner, weights=similarity, minlength=len(query))
        return total, total / np.maximum(n, 1)

    def _fit_embeddings(self, train, user_codes, video_codes, liked, shared):
        # A fixed early-history basis avoids current-label and future-event leakage.
        cutoff_index = min(len(train) - 1, max(0, int(len(train) * self.warmup_fraction) - 1))
        self.embedding_cutoff = pd.to_datetime(train.timestamp.iloc[cutoff_index], utc=True)
        warm = (pd.to_datetime(train.timestamp, utc=True) <= self.embedding_cutoff).to_numpy()
        # Implicit-feedback strength: every play counts; likes and shares add confidence.
        strength = (1.0 + liked + shared)[warm].astype(np.float32)
        matrix = csr_matrix(
            (strength, (user_codes[warm], video_codes[warm])),
            shape=(len(self.user_ids), len(self.video_ids)),
        )
        components = min(self.dimensions, min(matrix.shape) - 1)
        if components > 0 and matrix.nnz:
            svd = TruncatedSVD(n_components=components, random_state=self.seed)
            u = svd.fit_transform(matrix)
            v = svd.components_.T
        else:
            svd, components = None, 0
            u = np.zeros((len(self.user_ids), 0))
            v = np.zeros((len(self.video_ids), 0))
        u = np.pad(u, ((0, 0), (0, self.dimensions - components))).astype(np.float32)
        v = np.pad(v, ((0, 0), (0, self.dimensions - components))).astype(np.float32)
        self.embeddings = {
            "user_vectors": u,
            "video_vectors": v,
            "user_map": {value: n for n, value in enumerate(self.user_ids)},
            "video_map": {value: n for n, value in enumerate(self.video_ids)},
            "svd_model": svd,
        }

    # ------------------------------------------------------------- aggregation
    def _aggregate(self, index, codes, query, prefix_name, sessions=False):
        n, start = index.lookup(codes, query)
        valid = n > 0
        safe_n = np.maximum(n, 1)
        end = start + n
        last = np.maximum(end - 1, 0)
        sums, squares = index.total("watch", start, n), index.total("square", start, n)
        liked, shared = index.total("liked", start, n), index.total("shared", start, n)
        out = {
            "watch_mean": sums / safe_n,
            "watch_std": np.sqrt(
                np.maximum(0, (squares - sums**2 / safe_n) / np.maximum(n - 1, 1))
            ),
            "watch_count": n,
            "liked_mean": liked / safe_n,
            "liked_sum": liked,
            "shared_mean": shared / safe_n,
            "shared_sum": shared,
            "recency_days": np.where(valid, (query - index.time_at(last)) / NS_PER_DAY, 0),
            "tenure_days": np.where(valid, (query - index.time_at(start)) / NS_PER_DAY, 0),
        }
        if sessions:
            same = valid & ((query - index.time_at(last)) <= SESSION_GAP_NS)
            session_start = index.session_start[np.clip(last, 0, len(index.times) - 1)]
            older = n - np.minimum(n, 5)
            out.update(
                session_id=np.where(valid, index.session_id[last] + (~same), 0),
                session_length=np.where(same, end - session_start + 1, 1),
                session_duration_minutes=np.where(
                    same, (query - index.time_at(session_start)) / NS_PER_MINUTE, 0
                ),
                recent_engagement=(
                    (liked - index.total("liked", start, older))
                    + (shared - index.total("shared", start, older))
                )
                / np.maximum(np.minimum(n, 5), 1),
                category_diversity=index.total("new_category", start, n),
            )
        return {prefix_name + key: value for key, value in out.items()}, n, start

    # ------------------------------------------------------------- transform
    def transform(self, requests):
        frame = requests[["user_id", "video_id", "watch_time", "timestamp"]].reset_index(drop=True)
        watch = frame.watch_time.to_numpy(float)
        if not np.isfinite(watch).all() or (watch < 0).any():
            raise ValueError("watch_time must be finite and nonnegative")
        query = timestamps(frame.timestamp)
        hour = (query // NS_PER_HOUR) % 24
        day = (query // NS_PER_DAY + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
        month = query.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64) % 12 + 1
        user_codes = self.user_ids.get_indexer(frame.user_id)
        video_codes = self.video_ids.get_indexer(frame.video_id)
        known_user, known_video = user_codes >= 0, video_codes >= 0
        safe_user = np.where(known_user, user_codes, 0)
        safe_video = np.where(known_video, video_codes, 0)
        duration = np.where(known_video, self.video_duration[safe_video], self.default_duration)
        duration = np.clip(duration, 1e-6, None)
        category = np.where(known_video, self.video_category[safe_video], -1)
        creator = np.where(known_video, self.video_creator[safe_video], -1)
        upload = np.where(known_video, self.video_upload[safe_video], query)
        scaler = self.scalers["watch_time"]

        features = {
            "watch_time": watch,
            "duration_sec": duration,
            "hour_of_day": hour,
            "day_of_week": day,
            "is_weekend": (day >= 5).astype(int),
            "month": month,
            "completion_rate": np.clip(watch / duration, 0, 1),
            # Same arithmetic as MinMaxScaler.transform, without per-call input validation.
            "watch_time_normalized": watch * scaler.scale_[0] + scaler.min_[0],
            "hour_weekend_interaction": hour * (day >= 5),
            "user_cold_start": None,
            "video_cold_start": None,
            "video_age_days": np.clip((query - upload) / NS_PER_DAY, 0, None),
        }
        for column in USER_NUMERIC:
            values = np.where(
                known_user, self.user_profile[column][safe_user], self.profile_defaults[column]
            )
            features[column] = np.log1p(values) if column in COUNT_COLUMNS else values

        user, user_n, user_start = self._aggregate(
            self.user_index, user_codes, query, "user_", sessions=True
        )
        video, video_n, video_start = self._aggregate(
            self.video_index, video_codes, query, "video_"
        )
        features.update(user)
        features.update(video)
        features["user_cold_start"] = (user_n == 0).astype(int)
        features["video_cold_start"] = (video_n == 0).astype(int)
        features["recency_days"] = features["user_recency_days"]
        features["interactions_per_day"] = user_n / (features["user_tenure_days"] + 1)
        features["video_popularity"] = video_n
        features["engagement_rate"] = (
            features["video_liked_sum"] + features["video_shared_sum"]
        ) / np.maximum(video_n, 1)
        features["popularity_watch_ratio"] = video_n / (watch + 1)

        # Watch history of this user in the candidate's category and with its creator.
        alpha, priors = self.smoothing, self.priors
        n_categories = len(self.categories)
        pair = np.where(known_user & (category >= 0), user_codes * n_categories + category, -1)
        cat_n, cat_start = self.user_category_index.lookup(pair, query)
        cat_index = self.user_category_index
        cat_decay = cat_index.total("decay", cat_start, cat_n)
        all_decay = self.user_index.total("decay", user_start, user_n)
        creator_key = np.where(
            known_user & (creator >= 0), user_codes * len(self.creator_ids) + creator, -1
        )
        creator_code = np.where(
            creator_key >= 0, self.user_creator_pairs.get_indexer(creator_key), -1
        )
        creator_n, creator_start = self.user_creator_index.lookup(creator_code, query)
        creator_decay = self.user_creator_index.total("decay", creator_start, creator_n)
        # Plays of the video in the 24 hours before the request (snapshot at the cutoff).
        frozen = np.minimum(query, self.cutoff_ns + 1)
        recent_n = self.video_index.lookup(video_codes, frozen - NS_PER_DAY)[0]
        visible_n = self.video_index.lookup(video_codes, frozen)[0]
        size = len(frame)
        features.update(
            user_cat_watch_count=cat_n,
            user_cat_like_rate=(
                cat_index.total("liked", cat_start, cat_n) + alpha * priors["like_rate"]
            )
            / (cat_n + alpha),
            user_cat_completion=(
                cat_index.total("completion", cat_start, cat_n) + alpha * priors["completion"]
            )
            / (cat_n + alpha),
            user_cat_decayed_share=np.divide(
                cat_decay, all_decay, out=np.zeros(size), where=all_decay > 0
            ),
            user_cat_decayed_completion=np.divide(
                cat_index.total("decay_completion", cat_start, cat_n),
                cat_decay,
                out=np.full(size, priors["completion"]),
                where=cat_decay > 0,
            ),
            user_creator_watch_count=creator_n,
            user_creator_decayed_share=np.divide(
                creator_decay, all_decay, out=np.zeros(size), where=all_decay > 0
            ),
            user_completion_mean=(
                self.user_index.total("completion", user_start, user_n)
                + alpha * priors["completion"]
            )
            / (user_n + alpha),
            user_long_view_rate=(
                self.user_index.total("long_view", user_start, user_n) + alpha * priors["long_view"]
            )
            / (user_n + alpha),
            video_completion_mean=(
                self.video_index.total("completion", video_start, video_n)
                + alpha * priors["completion"]
            )
            / (video_n + alpha),
            video_long_view_rate=(
                self.video_index.total("long_view", video_start, video_n)
                + alpha * priors["long_view"]
            )
            / (video_n + alpha),
            video_plays_24h=visible_n - recent_n,
        )
        features["cowatch_score"], features["cowatch_mean"] = self.cowatch_scores(
            user_codes, video_codes, query
        )

        def column_values(values, codes, known):
            names = np.asarray(values, dtype=object)
            return np.where(known, names[np.maximum(codes, 0)], "unknown")

        categorical = {
            "user_active_degree": np.where(known_user, self.user_activity[safe_user], "UNKNOWN"),
            "category": column_values(self.categories, category, category >= 0),
            "upload_type": np.where(
                known_video, self.videos.upload_type.to_numpy(object)[safe_video], "unknown"
            ),
            "video_type": np.where(
                known_video, self.videos.video_type.to_numpy(object)[safe_video], "unknown"
            ),
            "day_period": np.select(
                [hour < 6, hour < 12, hour < 18], ["night", "morning", "afternoon"], "evening"
            ),
        }
        encoder = self.encoders["categorical"]
        for field, values in zip(encoder.feature_names_in_, encoder.categories_):
            for value in values:
                features[f"{field}_{value}"] = categorical[field] == value

        available = query > self.embedding_cutoff.value
        embedded = {}
        for kind, codes in [("user", user_codes), ("video", video_codes)]:
            vectors = self.embeddings[kind + "_vectors"][np.maximum(codes, 0)].copy()
            vectors[(codes < 0) | ~available] = 0
            embedded[kind] = vectors
            for n in range(self.dimensions):
                features[f"{kind}_emb_{n}"] = vectors[:, n]
        features["svd_affinity"] = (embedded["user"] * embedded["video"]).sum(axis=1)

        ordered = self._column_order(features)
        matrix = np.empty((len(frame), len(ordered)), dtype=np.float32)
        for column, name in enumerate(ordered):
            matrix[:, column] = features[name]
        np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(matrix, columns=ordered)

    def transform_in_chunks(self, requests, chunk_size=200_000):
        """``transform`` for large candidate sets with bounded temporary memory."""
        parts = [
            self.transform(requests.iloc[start : start + chunk_size])
            for start in range(0, len(requests), chunk_size)
        ]
        return pd.concat(parts, ignore_index=True) if parts else self.transform(requests)

    def _column_order(self, features):
        if getattr(self, "feature_columns", None):
            return self.feature_columns
        embedding = [n for n in features if "_emb_" in n] + ["svd_affinity"]
        rest = [n for n in features if n not in embedding and n not in WATCH_HISTORY_FEATURES]
        return rest + WATCH_HISTORY_FEATURES + embedding

    # ---------------------------------------------------------- serving helpers
    def video_history_counts(self):
        return self.video_index.counts.copy()

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
