import pytest
from pandera.errors import SchemaErrors

from src.data_validation import load_data


def write_data(root, sample_data):
    for name, frame in zip(["users.csv", "videos.csv", "interactions_cleaned.csv"], sample_data):
        frame.to_csv(root / name, index=False)


def test_valid_tables(tmp_path, sample_data):
    write_data(tmp_path, sample_data)
    result = load_data(tmp_path)
    assert len(result.interactions) == 100
    assert str(result.interactions.timestamp.dt.tz) == "UTC"


@pytest.mark.parametrize(
    "case", ["duplicate", "foreign_key", "invalid_duration", "invalid_age", "infinite_watch"]
)
def test_invalid_tables_rejected(tmp_path, sample_data, case):
    users, videos, events = [frame.copy() for frame in sample_data]
    if case == "duplicate":
        events.loc[1] = events.loc[0]
    elif case == "foreign_key":
        events.loc[0, "video_id"] = "video_9999"
    elif case == "invalid_duration":
        videos.loc[0, "duration_sec"] = -1.0
    elif case == "invalid_age":
        users.loc[0, "age"] = 8
    else:
        events.loc[0, "watch_time"] = float("inf")
    write_data(tmp_path, (users, videos, events))
    with pytest.raises((ValueError, SchemaErrors)):
        load_data(tmp_path)
