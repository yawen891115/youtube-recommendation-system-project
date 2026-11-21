import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from src.feature_engineering import FeatureEngineer, timestamps
from src.ranking import (
    SVDRecommender,
    bootstrap_ci,
    build_candidates,
    feature_group,
    first_plays,
    group_sizes,
    grouped_importance,
    item_knn_scores,
    paired_bootstrap_test,
    per_query_metrics,
    ranking_columns,
)
from src.train_ranker import fit_lambdarank, validation_ndcg


def test_ndcg_matches_scikit_learn():
    rng = np.random.default_rng(7)
    sizes = [5, 12, 30, 8]
    labels = np.concatenate([rng.integers(0, 3, size) for size in sizes])
    labels[0] = 1  # every query has at least one relevant item
    labels[5] = labels[17] = labels[47] = 2
    scores = rng.random(len(labels))
    queries = np.repeat(np.arange(len(sizes)), sizes)
    ours = per_query_metrics(scores, labels, queries)["ndcg@10"].to_numpy()
    expected = [
        ndcg_score([labels[queries == q]], [scores[queries == q]], k=10) for q in range(len(sizes))
    ]
    np.testing.assert_allclose(ours, expected)


def test_single_positive_metrics_by_hand():
    # The relevant item has the 3rd highest of 5 scores.
    metrics = per_query_metrics([0.9, 0.8, 0.5, 0.2, 0.1], [0, 0, 1, 0, 0], [0] * 5).iloc[0]
    assert metrics["ndcg@10"] == pytest.approx(1 / np.log2(4))
    assert metrics["hr@10"] == 1
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["auc"] == pytest.approx(0.5)


def test_queries_without_relevant_items_are_skipped_and_groups_are_contiguous():
    metrics = per_query_metrics([1, 2, 3, 4], [0, 0, 1, 0], [3, 3, 9, 9])
    assert len(metrics) == 1
    assert group_sizes([5, 5, 5, 2, 2, 7]).tolist() == [3, 2, 1]


def candidate_inputs(n_users=40, n_videos=200, seed=0):
    rng = np.random.default_rng(seed)
    videos = np.array([f"video_{n}" for n in range(n_videos)])
    events = pd.DataFrame(
        {
            "user_id": [f"user_{n % n_users}" for n in range(300)],
            "video_id": rng.choice(videos, 300),
            "liked": (np.arange(300) % 3 == 0).astype(int),
            "timestamp": pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        }
    )
    # Half of the catalog only appears on day 3, after many plays have happened.
    uploads = np.where(
        np.arange(n_videos) < n_videos // 2,
        pd.Timestamp("2023-12-01", tz="UTC").value,
        pd.Timestamp("2024-01-03", tz="UTC").value,
    )
    interacted = events.groupby("user_id").video_id.agg(set).to_dict()
    return events, videos, uploads, interacted


def test_candidates_hold_one_positive_and_eligible_unseen_negatives():
    events, catalog, uploads, interacted = candidate_inputs()
    candidates = build_candidates(events, catalog, uploads, interacted, n_negatives=20, seed=1)
    assert candidates.query_id.nunique() == len(events)
    available = dict(zip(catalog, uploads))
    for _, query in candidates.groupby("query_id"):
        assert len(query) == 21 and query.label.sum() == 1
        assert query.video_id.is_unique and query.timestamp.nunique() == 1
        negatives = query[query.label == 0]
        assert not negatives.video_id.isin(interacted[query.user_id.iloc[0]]).any()
        uploaded = negatives.video_id.map(available).to_numpy()
        assert (uploaded <= query.timestamp.iloc[0].value).all()


def test_first_plays_skip_rewatches():
    plays = pd.DataFrame(
        {
            "user_id": ["user_1", "user_1", "user_2", "user_1"],
            "video_id": ["video_1"] * 3 + ["video_2"],
        }
    )
    assert first_plays(plays).tolist() == [True, False, True, True]


def test_score_ties_are_not_broken_in_favour_of_the_positive():
    events, catalog, uploads, interacted = candidate_inputs()
    candidates = build_candidates(events, catalog, uploads, interacted, n_negatives=99, seed=2)
    metrics = per_query_metrics(np.zeros(len(candidates)), candidates.label, candidates.query_id)
    # With all scores tied the positive's rank is random: HR@10 near 10/100, never always 1.
    assert 0.0 < metrics["hr@10"].mean() < 0.3
    positions = candidates.groupby("query_id").label.apply(lambda s: int(np.argmax(s.to_numpy())))
    assert positions.nunique() > 10


def test_bootstrap_interval_brackets_the_mean():
    values = np.random.default_rng(3).random(500)
    ci = bootstrap_ci(values, seed=4)
    assert ci["low"] < ci["mean"] < ci["high"]
    assert ci == bootstrap_ci(values, seed=4)


def test_paired_bootstrap_detects_a_real_gain_and_not_noise():
    rng = np.random.default_rng(5)
    baseline = rng.random(3000)
    better = baseline + 0.02 + rng.normal(0, 0.05, 3000)
    gain = paired_bootstrap_test(better, baseline, n_resamples=2000, seed=6)
    assert gain["low"] > 0 and gain["p_value"] < 0.01
    noise = paired_bootstrap_test(baseline + rng.normal(0, 0.05, 3000), baseline, n_resamples=2000)
    assert noise["low"] < 0 < noise["high"] and noise["p_value"] > 0.05


def test_feature_families():
    assert feature_group("video_emb_3") == "SVD embeddings"
    assert feature_group("svd_affinity") == "SVD embeddings"
    assert feature_group("user_cat_decayed_share") == "Category & creator watch history"
    assert feature_group("user_creator_watch_count") == "Category & creator watch history"
    assert feature_group("user_watch_mean") == "User watch history"
    assert feature_group("user_active_degree_full_active") == "User profile"
    assert feature_group("is_video_author") == "User profile"
    assert feature_group("category_tag_3") == "Video metadata"
    assert feature_group("video_age_days") == "Video metadata"
    assert feature_group("video_popularity") == "Video history & popularity"
    assert feature_group("cowatch_score") == "Co-watch history"
    assert feature_group("is_weekend") == "Request context"
    shares = grouped_importance(["video_emb_1", "register_days", "hour_of_day"], [2.0, 1.0, 1.0])
    assert shares[0] == {"group": "SVD embeddings", "share": 0.5}
    assert sum(row["share"] for row in shares) == pytest.approx(1)


def taste_data(seed=0, n_users=60, n_videos=240, n_events=3000):
    """Synthetic plays where each user favours one category, in the project schema."""
    rng = np.random.default_rng(seed)
    categories = [f"tag_{n}" for n in range(8)]
    users = pd.DataFrame(
        {
            "user_id": [f"user_{n}" for n in range(n_users)],
            "user_active_degree": "full_active",
            "is_lowactive_period": 0,
            "is_live_streamer": 0,
            "is_video_author": 0,
            "follow_user_num": rng.integers(0, 50, n_users),
            "fans_user_num": rng.integers(0, 50, n_users),
            "friend_user_num": rng.integers(0, 5, n_users),
            "register_days": rng.integers(30, 900, n_users),
        }
    )
    videos = pd.DataFrame(
        {
            "video_id": [f"video_{n}" for n in range(n_videos)],
            "creator_id": np.arange(n_videos) // 4,
            "category": rng.choice(categories, n_videos),
            "duration_sec": rng.uniform(15, 160, n_videos),
            "video_type": "NORMAL",
            "upload_type": "Web",
            "upload_date": pd.Timestamp("2023-01-01", tz="UTC"),
        }
    )
    taste = rng.choice(categories, n_users)
    by_category = {name: videos.video_id[videos.category == name].to_numpy() for name in categories}
    owners = rng.integers(0, n_users, n_events)
    picks = [
        rng.choice(by_category[taste[u]]) if rng.random() < 0.7 else rng.choice(videos.video_id)
        for u in owners
    ]
    events = pd.DataFrame(
        {
            "user_id": [f"user_{u}" for u in owners],
            "video_id": picks,
            "timestamp": pd.date_range("2024-01-01", periods=n_events, freq="37min", tz="UTC"),
            "watch_time": rng.uniform(1, 30, n_events),
            "liked": (rng.random(n_events) < 0.1).astype(int),
            "shared": 0,
            "long_view": (rng.random(n_events) < 0.5).astype(int),
        }
    )
    return users, videos, events


def test_lambdarank_ranker_learns_user_taste_from_watch_history():
    users, videos, events = taste_data()
    train, validation = events.iloc[:2000], events.iloc[2000:]
    engineer = FeatureEngineer(dimensions=4).fit(train, users, videos, history=events)
    interacted = events.groupby("user_id").video_id.agg(set).to_dict()
    uploads = timestamps(videos.upload_date)
    data = {}
    for offset, (name, part, negatives) in enumerate(
        [("train", train, 10), ("validation", validation, 30)]
    ):
        part = part[first_plays(events)[part.index]]
        candidates = build_candidates(part, videos.video_id, uploads, interacted, negatives, offset)
        data[name] = {
            "candidates": candidates,
            "features": engineer.transform(candidates.assign(watch_time=0.0)),
            "labels": candidates.label.to_numpy(),
            "queries": candidates.query_id.to_numpy(),
        }
    columns = ranking_columns(engineer.feature_columns)
    params = {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10}
    model = fit_lambdarank(params, data, columns, seed=0, threads=1)
    learned = validation_ndcg(model.predict(data["validation"]["features"][columns]), data)
    chance = validation_ndcg(np.zeros(len(data["validation"]["labels"])), data)
    knn = validation_ndcg(item_knn_scores(engineer, data["validation"]["candidates"]), data)
    assert learned > chance + 0.2 and knn > chance


def test_svd_baseline_scores_known_pairs_only():
    users, videos, events = taste_data(n_events=500)
    model = SVDRecommender(dimensions=3).fit(events, users.user_id, videos.video_id)
    candidates = pd.DataFrame(
        {"user_id": ["user_1", "user_999999"], "video_id": ["video_1", "video_1"]}
    )
    scores = model.score(candidates)
    assert np.isfinite(scores).all() and scores[1] == 0
