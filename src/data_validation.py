"""Validate the supplied user, video, and interaction tables."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

CATEGORIES = ["Comedy", "Education", "Fashion", "Food", "Gaming", "Music", "Sports", "Travel"]
USERS_SCHEMA = DataFrameSchema(
    {
        "user_id": Column(str, Check.str_matches(r"^user_\d+$"), unique=True),
        "age": Column(int, Check.in_range(13, 99)),
        "gender": Column(str, Check.isin(["M", "F", "Other"])),
        "preferred_category": Column(str, Check.isin(CATEGORIES)),
        "join_date": Column(str),
    },
    coerce=True,
)
VIDEOS_SCHEMA = DataFrameSchema(
    {
        "video_id": Column(str, Check.str_matches(r"^video_\d+$"), unique=True),
        "creator_id": Column(int, Check.ge(0)),
        "category": Column(str, Check.isin(CATEGORIES)),
        "duration_sec": Column(float, Check.gt(0)),
        "view_count": Column(int, Check.ge(0)),
        "upload_date": Column(str),
    },
    coerce=True,
)
INTERACTIONS_SCHEMA = DataFrameSchema(
    {
        "user_id": Column(str, Check.str_matches(r"^user_\d+$")),
        "video_id": Column(str, Check.str_matches(r"^video_\d+$")),
        "watch_time": Column(
            float, [Check.ge(0), Check(lambda s: s.map(lambda x: x < float("inf")).all())]
        ),
        "liked": Column(int, Check.isin([0, 1])),
        "shared": Column(int, Check.isin([0, 1])),
        "timestamp": Column(str),
    },
    coerce=True,
)


@dataclass
class DataBundle:
    users: pd.DataFrame
    videos: pd.DataFrame
    interactions: pd.DataFrame


def load_data(data_dir="data"):
    root = Path(data_dir)
    frames = []
    for file, schema in [
        ("users.csv", USERS_SCHEMA),
        ("videos.csv", VIDEOS_SCHEMA),
        ("interactions_cleaned.csv", INTERACTIONS_SCHEMA),
    ]:
        frame = schema.validate(pd.read_csv(root / file), lazy=True)
        if frame.duplicated().any():
            raise ValueError(f"{file} contains duplicate rows")
        frames.append(frame)
    users, videos, interactions = frames
    for frame, column in [
        (users, "join_date"),
        (videos, "upload_date"),
        (interactions, "timestamp"),
    ]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not interactions.user_id.isin(users.user_id).all():
        raise ValueError("Interactions contain unknown user IDs")
    if not interactions.video_id.isin(videos.video_id).all():
        raise ValueError("Interactions contain unknown video IDs")
    interactions = interactions.sort_values(
        ["timestamp", "user_id", "video_id"], kind="stable"
    ).reset_index(drop=True)
    return DataBundle(users, videos, interactions)


def describe_data(bundle):
    interactions = bundle.interactions
    merged = interactions.merge(
        bundle.videos[["video_id", "duration_sec", "upload_date"]], on="video_id"
    )
    return {
        "users": len(bundle.users),
        "videos": len(bundle.videos),
        "interactions": len(interactions),
        "positive_rate": float(interactions.liked.mean()),
        "share_rate": float(interactions.shared.mean()),
        "mean_watch_seconds": float(interactions.watch_time.mean()),
        "first_interaction": interactions.timestamp.min().isoformat(),
        "last_interaction": interactions.timestamp.max().isoformat(),
        "rewatch_rows": int((merged.watch_time > merged.duration_sec).sum()),
        "interactions_before_upload": int((merged.timestamp < merged.upload_date).sum()),
        "category_counts": bundle.videos.category.value_counts().to_dict(),
        "missing_cells": {
            name: int(frame.isna().sum().sum()) for name, frame in vars(bundle).items()
        },
    }


def chronological_split(frame):
    ordered = frame.sort_values(["timestamp", "user_id", "video_id"], kind="stable").reset_index(
        drop=True
    )
    first, second = int(len(ordered) * 0.64), int(len(ordered) * 0.80)
    if min(first, second - first, len(ordered) - second) < 2:
        raise ValueError("Not enough examples for a chronological 64/16/20 split")
    # Keep equal timestamps on one side of each boundary.
    times = ordered.timestamp.to_numpy()
    first = int(times.searchsorted(times[first], side="left"))
    second = int(times.searchsorted(times[second], side="left"))
    splits = {
        "train": ordered.iloc[:first].copy(),
        "validation": ordered.iloc[first:second].copy(),
        "test": ordered.iloc[second:].copy(),
    }
    for name, split in splits.items():
        if split.empty or split.liked.nunique() < 2:
            raise ValueError(
                f"{name} needs both label classes; do not shuffle future events into training"
            )
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    print(json.dumps(describe_data(load_data(args.data_dir)), indent=2))
