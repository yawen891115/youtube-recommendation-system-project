import numpy as np
import pandas as pd
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
    rows = trained_service.recommend("user_1", 5, "2025-01-01T00:00:00Z")
    assert rows
    seen = trained_service.engineer.seen["user_1"]
    assert not any(row["video_id"] in seen for row in rows)
    assert all(rows[index]["score"] >= rows[index + 1]["score"] for index in range(len(rows) - 1))
