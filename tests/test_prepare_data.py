import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_validation import load_data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_kuairand.py"
spec = importlib.util.spec_from_file_location("prepare_kuairand", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def write_raw(root, n_users=30, plays_per_user=12):
    """Tiny files with the KuaiRand-Pure columns the build reads."""
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2022-04-09T04:00:00Z").value // 10**6
    for user in range(n_users):
        for play in range(plays_per_user):
            rows.append(
                {
                    "user_id": user,
                    "video_id": int(rng.integers(0, 40)),
                    "time_ms": start + (user * plays_per_user + play) * 60_000,
                    "is_click": int(play % 4 != 3),
                    "is_like": int(play % 5 == 0),
                    "is_forward": int(play % 11 == 0),
                    "long_view": int(play % 2 == 0),
                    "play_time_ms": 12_000,
                    "duration_ms": 30_000,
                }
            )
    log = pd.DataFrame(rows)
    log = pd.concat([log, log.iloc[:5]], ignore_index=True)  # duplicate log rows
    half = len(log) // 2
    log.iloc[:half].to_csv(root / prepare.LOGS[0], index=False)
    log.iloc[half:].to_csv(root / prepare.LOGS[1], index=False)
    pd.DataFrame(
        {
            "user_id": range(n_users),
            "user_active_degree": "full_active",
            "is_lowactive_period": 0,
            "is_live_streamer": -124,
            "is_video_author": 1,
            "follow_user_num": 3,
            "fans_user_num": 4,
            "friend_user_num": 1,
            "register_days": 100,
        }
    ).to_csv(root / "user_features_pure.csv", index=False)
    pd.DataFrame(
        {
            "video_id": range(40),
            "author_id": np.arange(40) // 3,
            "video_type": "NORMAL",
            "upload_dt": "2022-04-08",
            "upload_type": "Web",
            "video_duration": 30_000.0,
            "tag": ["5,7"] * 39 + [np.nan],
        }
    ).to_csv(root / "video_features_basic_pure.csv", index=False)
    return log


def test_build_keeps_the_earliest_plays_of_random_users_once(tmp_path):
    log = write_raw(tmp_path)
    users, videos, plays = prepare.build(tmp_path, size=100, seed=42)
    assert len(plays) == 100 and not plays.duplicated().any()
    assert plays.timestamp.is_monotonic_increasing
    clicks = log[log.is_click == 1].drop_duplicates(["user_id", "video_id", "time_ms"])
    drawn = clicks[clicks.user_id.isin(plays.user_id.str[5:].astype(int))]
    assert len(plays) == min(100, len(drawn))
    assert set(users.user_id) == set(plays.user_id)
    assert videos.category.iloc[0] == "tag_5" and videos.category.iloc[-1] == "tag_unknown"
    assert users.is_live_streamer.eq(0).all() and users.is_video_author.eq(1).all()
    assert prepare.build(tmp_path, size=100, seed=42)[2].equals(plays)


def test_built_tables_pass_project_validation(tmp_path):
    write_raw(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    for name, frame in zip(
        ["users.csv", "videos.csv", "interactions_cleaned.csv"],
        prepare.build(tmp_path, size=120, seed=1),
    ):
        frame.to_csv(output / name, index=False)
    bundle = load_data(output)
    assert len(bundle.interactions) == 120
    assert (bundle.interactions.watch_time == 12.0).all()
