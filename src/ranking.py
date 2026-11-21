"""Pre-watch ranking protocol: candidate lists, ranking metrics, baselines and significance.

Protocol (fixed before any model was compared):
    * Chronological 64K / 16K / 20K split of the 100K plays (``chronological_split``).
    * One query per first-time play (the user had never played that video before): the played
      video plus ``N_EVAL_NEGATIVES`` (99) videos drawn uniformly from the catalog among videos
      that were already uploaded at the query time and that the user never played.
    * Every candidate feature is computed at the query timestamp from earlier plays only and
      never includes the current watch (``OBSERVED_FEATURES`` are removed).
    * Rows are shuffled inside each query so score ties are broken at random, never in favour
      of the positive.
    * Metrics are averaged over queries: NDCG@10 (primary), HR@10, MRR and per-query AUC.
      Model comparisons use a paired bootstrap over test queries.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .feature_engineering import OBSERVED_FEATURES, timestamps

N_EVAL_NEGATIVES = 99
TOP_K = 10


# --------------------------------------------------------------------------- candidates
def first_plays(events):
    """Boolean mask: the play is the user's first play of that video (events in time order)."""
    return ~events.duplicated(["user_id", "video_id"]).to_numpy()


def build_candidates(events, catalog, upload_times, interacted, n_negatives, seed):
    """One candidate list per event: the played video plus sampled unplayed videos.

    events: plays with user_id, video_id, timestamp (each becomes one query).
    catalog / upload_times: every recommendable video and when it became available.
    interacted: mapping user_id -> set of video ids the user played at any time; excluded
        from negatives so no unobserved positive is labelled irrelevant.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(np.asarray(upload_times), kind="stable")
    catalog = np.asarray(catalog)[order]
    available = np.searchsorted(
        np.asarray(upload_times)[order], timestamps(events.timestamp), side="right"
    )
    query_ids, users, videos, times, labels = [], [], [], [], []
    for query, row in enumerate(
        events[["user_id", "video_id", "timestamp"]].itertuples(index=False)
    ):
        pool = int(available[query])
        taken = set(interacted.get(row.user_id, ())) | {row.video_id}
        chosen = []
        for _ in range(50):
            if len(chosen) == n_negatives:
                break
            draw = rng.choice(pool, size=min(pool, n_negatives + 16), replace=False)
            for video in catalog[draw]:
                if video not in taken and len(chosen) < n_negatives:
                    taken.add(video)
                    chosen.append(video)
        size = len(chosen) + 1
        query_ids.extend([query] * size)
        users.extend([row.user_id] * size)
        videos.extend([row.video_id, *chosen])
        times.extend([row.timestamp] * size)
        labels.extend([1] + [0] * len(chosen))
    frame = pd.DataFrame(
        {
            "query_id": np.asarray(query_ids, dtype=np.int64),
            "user_id": users,
            "video_id": videos,
            "timestamp": pd.to_datetime(pd.Series(times), utc=True),
            "label": np.asarray(labels, dtype=np.int8),
        }
    )
    # Random order inside each query: stable sorting then breaks ties at random.
    frame["_order"] = rng.random(len(frame))
    frame = frame.sort_values(["query_id", "_order"], kind="stable").drop(columns="_order")
    return frame.reset_index(drop=True)


def group_sizes(query_ids):
    """Sizes of contiguous query groups, as LightGBM's ``group`` argument expects."""
    query_ids = np.asarray(query_ids)
    if len(query_ids) == 0:
        return np.zeros(0, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(query_ids)) + 1
    return np.diff(np.concatenate([[0], boundaries, [len(query_ids)]]))


def ranking_columns(feature_columns):
    return [column for column in feature_columns if column not in OBSERVED_FEATURES]


# ------------------------------------------------------------------------------ metrics
def per_query_metrics(scores, labels, query_ids, k=TOP_K):
    """Per-query NDCG@k, HR@k, MRR and AUC for contiguous query groups.

    Gains are linear in the label (identical to ``2**label - 1`` for binary labels, and to
    scikit-learn's ``ndcg_score``). Ties keep the incoming row order, which the candidate
    builder randomises. Queries without a relevant item are dropped.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    query_ids = np.asarray(query_ids)
    sizes = group_sizes(query_ids)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    group = np.repeat(np.arange(len(sizes)), sizes)
    position = np.arange(len(scores))
    order = np.lexsort((position, -scores, group))
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(len(scores)) - np.repeat(starts, sizes) + 1
    discount = 1.0 / np.log2(rank + 1)
    dcg = np.bincount(group, weights=labels * discount * (rank <= k), minlength=len(sizes))
    ideal_order = np.lexsort((position, -labels, group))
    ideal_rank = np.empty(len(scores), dtype=np.int64)
    ideal_rank[ideal_order] = np.arange(len(scores)) - np.repeat(starts, sizes) + 1
    idcg = np.bincount(
        group,
        weights=labels / np.log2(ideal_rank + 1) * (ideal_rank <= k),
        minlength=len(sizes),
    )
    relevant = labels > 0
    first_hit = np.full(len(sizes), np.inf)
    np.minimum.at(first_hit, group[relevant], rank[relevant])
    positives = np.bincount(group, weights=relevant, minlength=len(sizes))
    negatives = sizes - positives
    # AUC: for each relevant row, the share of irrelevant rows scored below it (ties = 0.5).
    below = np.zeros(len(sizes))
    for g in np.flatnonzero(positives > 0):
        rows = slice(starts[g], starts[g] + sizes[g])
        neg_scores = np.sort(scores[rows][~relevant[rows]])
        pos_scores = scores[rows][relevant[rows]]
        lower = np.searchsorted(neg_scores, pos_scores, side="left")
        upper = np.searchsorted(neg_scores, pos_scores, side="right")
        below[g] = (lower + 0.5 * (upper - lower)).sum()
    keep = idcg > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(
            {
                "ndcg@10": dcg[keep] / idcg[keep],
                "hr@10": (first_hit[keep] <= k).astype(float),
                "mrr": 1.0 / first_hit[keep],
                "auc": below[keep] / (positives[keep] * np.maximum(negatives[keep], 1)),
            }
        )


def summarize(per_query):
    return {metric: float(per_query[metric].mean()) for metric in per_query.columns}


def _resampled_means(values, n_resamples, seed, batch=200):
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        index = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[index].mean(axis=1))
    return np.concatenate(means)


def bootstrap_ci(values, n_resamples=1000, seed=42, level=0.95):
    """Percentile bootstrap interval of a mean over queries."""
    values = np.asarray(values, dtype=np.float64)
    means = _resampled_means(values, n_resamples, seed)
    tail = (1 - level) / 2
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1 - tail)),
    }


def paired_bootstrap_test(first, second, n_resamples=10_000, seed=42, level=0.95):
    """Paired bootstrap over queries for mean(first - second).

    Both models are scored on the same queries; queries are resampled with replacement and
    the per-query differences are averaged. The two-sided p-value is twice the share of
    resampled mean differences on the far side of zero (floored at one resample).
    """
    differences = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    means = _resampled_means(differences, n_resamples, seed)
    tail = (1 - level) / 2
    beyond = min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_difference": float(differences.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1 - tail)),
        "p_value": float(max(2 * beyond, 1 / n_resamples)),
        "resamples": int(n_resamples),
        "queries": int(len(differences)),
    }


# ---------------------------------------------------------------------- reference scores
def score_model(model, features):
    """Higher score = rank earlier, for rankers and probabilistic classifiers alike."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def random_scores(candidates, seed):
    return np.random.default_rng(seed).random(len(candidates))


def popularity_scores(features):
    """Non-personalised: plays of the video in the training window (frozen snapshot)."""
    return features["video_popularity"].to_numpy(np.float64)


def trending_scores(features):
    """Non-personalised: plays in the last 24 hours before the request (snapshot at cutoff)."""
    return features["video_plays_24h"].to_numpy(np.float64)


def category_history_scores(features):
    """Content-based: recency-weighted share of the user's plays in the video's category."""
    return features["user_cat_decayed_share"].to_numpy(np.float64)


def item_knn_scores(engineer, candidates):
    """Item-based collaborative filtering: summed cosine similarity (training co-watch counts)
    between the candidate and every video the user played before the request."""
    total, _ = engineer.cowatch_scores(
        engineer.user_ids.get_indexer(candidates.user_id),
        engineer.video_ids.get_indexer(candidates.video_id),
        timestamps(candidates.timestamp),
        leave_user_out=False,
    )
    return total


def _interaction_matrix(interactions, user_index, video_index, binary=False):
    rows = user_index.get_indexer(interactions.user_id)
    cols = video_index.get_indexer(interactions.video_id)
    if binary:
        values = np.ones(len(rows))
    else:
        values = 1.0 + interactions.liked.to_numpy(float) + interactions.shared.to_numpy(float)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(user_index), len(video_index)))
    if binary:
        matrix.data = np.minimum(matrix.data, 1.0)
    return matrix


class SVDRecommender:
    """Collaborative filtering: rank-50 truncated SVD of the training user x video
    implicit-feedback matrix; score = reconstructed affinity u . v."""

    def __init__(self, dimensions=50, seed=42):
        self.dimensions, self.seed = dimensions, seed

    def fit(self, interactions, user_ids, video_ids):
        self.user_index, self.video_index = pd.Index(user_ids), pd.Index(video_ids)
        matrix = _interaction_matrix(interactions, self.user_index, self.video_index)
        svd = TruncatedSVD(n_components=self.dimensions, random_state=self.seed)
        self.user_factors = svd.fit_transform(matrix)
        self.video_factors = svd.components_.T
        return self

    def score(self, candidates):
        users = self.user_index.get_indexer(candidates.user_id)
        videos = self.video_index.get_indexer(candidates.video_id)
        scores = np.einsum(
            "ij,ij->i",
            self.user_factors[np.maximum(users, 0)],
            self.video_factors[np.maximum(videos, 0)],
        )
        return np.where((users >= 0) & (videos >= 0), scores, 0.0)


# ------------------------------------------------------------------- feature families
FEATURE_GROUPS = [
    ("SVD embeddings", lambda c: "_emb_" in c or c == "svd_affinity"),
    (
        "Category & creator watch history",
        lambda c: c.startswith(("user_cat_", "user_creator_")),
    ),
    ("Co-watch history", lambda c: c.startswith("cowatch_")),
    (
        "User watch history",
        lambda c: (
            c.startswith("user_")
            and not c.startswith("user_active_degree_")
            or c in {"recency_days", "interactions_per_day"}
        ),
    ),
    (
        "Video metadata",
        lambda c: (
            c in {"duration_sec", "video_age_days"}
            or c.startswith(("category_", "upload_type_", "video_type_"))
        ),
    ),
    (
        "Video history & popularity",
        lambda c: c.startswith("video_") or c == "engagement_rate",
    ),
    (
        "User profile",
        lambda c: (
            c.startswith("user_active_degree_")
            or c
            in {
                "register_days",
                "follow_user_num",
                "fans_user_num",
                "friend_user_num",
                "is_video_author",
                "is_live_streamer",
                "is_lowactive_period",
            }
        ),
    ),
    ("Request context", lambda c: True),
]


def feature_group(column):
    for name, member in FEATURE_GROUPS:
        if member(column):
            return name
    return "Request context"


def grouped_importance(columns, importance):
    totals = {}
    for column, value in zip(columns, importance):
        group = feature_group(column)
        totals[group] = totals.get(group, 0.0) + float(value)
    total = sum(totals.values()) or 1.0
    return sorted(
        ({"group": group, "share": value / total} for group, value in totals.items()),
        key=lambda row: -row["share"],
    )
