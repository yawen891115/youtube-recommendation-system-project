import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_validation import chronological_split
from src.feature_engineering import OBSERVED_FEATURES, FeatureEngineer


def test_chronological_partition(sample_data):
    _, _, events = sample_data
    split = chronological_split(events.sample(frac=1, random_state=42))
    assert [len(split[key]) for key in ["train", "validation", "test"]] == [64, 16, 20]
    assert split["train"].timestamp.max() < split["validation"].timestamp.min()
    assert split["validation"].timestamp.max() < split["test"].timestamp.min()


def test_tied_timestamps_stay_together(sample_data):
    _, _, events = sample_data
    events = events.copy()
    events.loc[63:65, "timestamp"] = events.loc[64, "timestamp"]
    split = chronological_split(events)
    assert split["train"].timestamp.max() < split["validation"].timestamp.min()


def test_current_and_future_labels_do_not_leak(sample_data):
    users, videos, events = sample_data
    first = FeatureEngineer(dimensions=5).fit(events, users, videos)
    changed = events.copy()
    changed.loc[40:, ["liked", "shared"]] = 1 - changed.loc[40:, ["liked", "shared"]]
    second = FeatureEngineer(dimensions=5).fit(changed, users, videos)
    assert_frame_equal(first.transform(events.iloc[20:41]), second.transform(events.iloc[20:41]))


def test_no_history_or_embedding_at_first_event(sample_data):
    users, videos, events = sample_data
    engineer = FeatureEngineer(dimensions=5).fit(events, users, videos)
    row = engineer.transform(events.iloc[:1]).iloc[0]
    assert row.user_watch_count == row.video_watch_count == 0
    assert row.user_cold_start == row.video_cold_start == 1
    assert row.user_session_length == 1
    assert (row.filter(like="_emb_") == 0).all()


def test_sessions_only_include_previous_events(sample_data):
    users, videos, events = sample_data
    engineer = FeatureEngineer(dimensions=5).fit(events, users, videos)
    row = engineer.transform(events.iloc[6:7]).iloc[0]
    assert row.user_watch_count == 2
    assert row.user_session_length == 3
    assert row.user_session_duration_minutes == 30


def test_cold_start_is_finite(sample_data):
    users, videos, events = sample_data
    engineer = FeatureEngineer(dimensions=5).fit(events, users, videos)
    request = pd.DataFrame(
        [
            {
                "user_id": "user_9999",
                "video_id": "video_9999",
                "watch_time": 0.0,
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ]
    )
    features = engineer.transform(request)
    assert np.isfinite(features.to_numpy()).all()
    assert features.user_cold_start.iloc[0] == 1
    assert features.video_cold_start.iloc[0] == 1
    assert features.user_session_length.iloc[0] == 1


def test_training_and_serving_agree(trained_service, sample_data):
    events = sample_data[2].iloc[80:84]
    features = trained_service.engineer.transform(events)[
        trained_service.metadata["feature_columns"]
    ]
    expected = trained_service.model.predict_proba(features)[:, 1]
    actual = [row["probability"] for row in trained_service.predict(events.to_dict("records"))]
    np.testing.assert_allclose(expected, actual)


def test_recommendations_exclude_seen_and_postwatch(trained_service):
    assert not set(OBSERVED_FEATURES) & set(trained_service.metadata["ranking_feature_columns"])
    rows = trained_service.recommend("user_1", 5, "2025-01-01T00:00:00Z")["recommendations"]
    assert rows
    seen = trained_service.engineer.seen["user_1"]
    assert not any(row["video_id"] in seen for row in rows)
    assert all(rows[index]["score"] >= rows[index + 1]["score"] for index in range(len(rows) - 1))


def test_watch_history_features_match_brute_force(sample_data):
    users, videos, events = sample_data
    engineer = FeatureEngineer(dimensions=5).fit(events.iloc[:64], users, videos)
    meta = videos.set_index("video_id")
    train = events.iloc[:64].join(meta[["category", "creator_id"]], on="video_id")
    scale = engineer.half_life_days * 86400 / np.log(2)
    requests = events.iloc[[10, 30, 50, 63, 70, 90]].copy()
    features = engineer.transform(requests)
    alpha, prior = engineer.smoothing, engineer.priors
    for (_, request), (_, row) in zip(requests.iterrows(), features.iterrows()):
        category, creator = meta.category[request.video_id], meta.creator_id[request.video_id]
        before = train[(train.user_id == request.user_id) & (train.timestamp < request.timestamp)]
        same = before[before.category == category]
        assert row.user_cat_watch_count == len(same)
        expected_like = (same.liked.sum() + alpha * prior["like_rate"]) / (len(same) + alpha)
        assert row.user_cat_like_rate == pytest.approx(expected_like, rel=1e-5)
        age = (request.timestamp - before.timestamp).dt.total_seconds().to_numpy()
        weights = np.exp(-age / scale)
        share = weights[(before.category == category).to_numpy()].sum() / weights.sum()
        assert row.user_cat_decayed_share == pytest.approx(share if len(before) else 0, rel=1e-5)
        assert row.user_creator_watch_count == (before.creator_id == creator).sum()
        times, now = train.timestamp.astype("int64"), request.timestamp.value
        window = min(now, times.max() + 1)
        recent = train[
            (train.video_id == request.video_id)
            & (times < window)
            & (times >= window - 86_400 * 10**9)
        ]
        assert row.video_plays_24h == len(recent)


def test_full_log_history_is_point_in_time(sample_data):
    users, videos, events = sample_data
    train = events.iloc[:64]
    full = FeatureEngineer(dimensions=5).fit(train, users, videos, history=events)
    frozen = FeatureEngineer(dimensions=5).fit(train, users, videos)
    late = events.iloc[[90]]
    # A request after the training cutoff also sees the later (non-training) plays ...
    assert (
        full.transform(late).user_watch_count.iloc[0]
        > frozen.transform(late).user_watch_count.iloc[0]
    )
    # ... but never plays at or after its own time, and learned inputs ignore them.
    changed = events.copy()
    changed.loc[85:, ["liked", "shared", "long_view"]] = (
        1 - changed.loc[85:, ["liked", "shared", "long_view"]]
    )
    other = FeatureEngineer(dimensions=5).fit(train, users, videos, history=changed)
    assert_frame_equal(full.transform(events.iloc[60:86]), other.transform(events.iloc[60:86]))
    assert full.priors == frozen.priors


def test_cowatch_similarity_leaves_the_user_out():
    users = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2", "user_3"],
            "user_active_degree": ["full_active"] * 3,
            **{column: [0] * 3 for column in ["is_lowactive_period", "is_live_streamer"]},
            **{column: [0] * 3 for column in ["is_video_author", "follow_user_num"]},
            **{column: [0] * 3 for column in ["fans_user_num", "friend_user_num"]},
            "register_days": [10] * 3,
        }
    )
    videos = pd.DataFrame(
        {
            "video_id": ["video_1", "video_2"],
            "creator_id": [0, 1],
            "category": ["tag_1", "tag_2"],
            "duration_sec": [30.0, 30.0],
            "video_type": ["NORMAL"] * 2,
            "upload_type": ["Web"] * 2,
            "upload_date": pd.to_datetime(["2023-01-01"] * 2, utc=True),
        }
    )
    # user_1 and user_2 play video_1 then video_2; user_3 plays only video_1.
    plays = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2", "user_3", "user_1", "user_2"],
            "video_id": ["video_1", "video_1", "video_1", "video_2", "video_2"],
            "watch_time": 10.0,
            "liked": 0,
            "shared": 0,
            "long_view": 0,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        }
    )
    engineer = FeatureEngineer(dimensions=1).fit(plays, users, videos)
    when = pd.Timestamp("2024-01-02", tz="UTC").value
    codes = engineer.user_ids.get_indexer(["user_1", "user_3"])
    video = engineer.video_ids.get_indexer(["video_2", "video_2"])
    query = np.array([when, when])
    left_out, _ = engineer.cowatch_scores(codes, video, query)
    standard, _ = engineer.cowatch_scores(codes, video, query, leave_user_out=False)
    # co-watch(1, 2) = 2, plays(1) = 3, plays(2) = 2; user_1's own pair is removed.
    np.testing.assert_allclose(left_out, [1 / np.sqrt(2), 1.0])
    np.testing.assert_allclose(standard, [2 / np.sqrt(6)] * 2)
