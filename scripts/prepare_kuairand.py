"""Build the project's 100,000-interaction dataset from KuaiRand-Pure (real Kuaishou logs).

Source: KuaiRand-Pure, Gao et al., "KuaiRand: An Unbiased Sequential Recommendation Dataset
with Randomly Exposed Videos", CIKM 2022, https://kuairand.com, licensed CC BY-SA 4.0.
Inputs: log_standard_4_08_to_4_21_pure.csv, log_standard_4_22_to_5_08_pure.csv,
user_features_pure.csv and video_features_basic_pure.csv.

Subset rules (fixed before any model was trained):
  * An interaction is a logged play: a standard-feed row with is_click == 1. Duplicate log rows
    of the same play (same user, video and millisecond) are kept once.
  * Users are drawn uniformly at random (seed 42) until their plays reach 100,000; every play
    of the drawn users is kept, and the earliest 100,000 plays by time form the dataset.
  * The catalog is every video in video_features_basic_pure.csv.

Output (project schema): users.csv, videos.csv, interactions_cleaned.csv.

Usage:
    python scripts/prepare_kuairand.py --download            # official Zenodo archive
    python scripts/prepare_kuairand.py --download --mirror github
    python scripts/prepare_kuairand.py --raw-dir path/to/KuaiRand-Pure/data
"""

import argparse
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LOGS = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]
FILES = LOGS + ["user_features_pure.csv", "video_features_basic_pure.csv"]
ZENODO = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
ZENODO_MD5 = "0820331067a3784d9691136f772b35a7"
GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/Applied-Machine-Learning-Lab/KuaiSim/2ae32aa/"
    "dataset/kuairand/kuairand-Pure/data/"
)
SOURCE_TIMEZONE = "Asia/Shanghai"  # upload_dt is a calendar date on a China-based platform
USER_COLUMNS = [
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]


def download(raw_dir, mirror):
    raw_dir.mkdir(parents=True, exist_ok=True)
    if mirror == "zenodo":
        payload = urllib.request.urlopen(ZENODO, timeout=600).read()
        if hashlib.md5(payload).hexdigest() != ZENODO_MD5:
            raise ValueError("KuaiRand-Pure archive checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = Path(member.name).name
                if name in FILES and member.isfile():
                    (raw_dir / name).write_bytes(archive.extractfile(member).read())
    else:
        for name in FILES:
            (raw_dir / name).write_bytes(
                urllib.request.urlopen(GITHUB_MIRROR + name, timeout=600).read()
            )
    missing = [name for name in FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing KuaiRand-Pure files: {missing}")


def primary_tag(value):
    if pd.isna(value) or str(value).strip() == "":
        return "tag_unknown"
    return f"tag_{str(value).split(',')[0].strip()}"


def build(raw_dir, size=100_000, seed=42):
    logs = pd.concat([pd.read_csv(raw_dir / name) for name in LOGS], ignore_index=True)
    plays = logs[logs.is_click == 1].drop_duplicates(["user_id", "video_id", "time_ms"])
    users = plays.user_id.drop_duplicates().to_numpy()
    order = np.random.default_rng(seed).permutation(len(users))
    counts = plays.user_id.value_counts()
    drawn = users[order]
    cumulative = counts.reindex(drawn).cumsum().to_numpy()
    selected = drawn[: int(np.searchsorted(cumulative, size)) + 1]
    sample = plays[plays.user_id.isin(selected)].sort_values(
        ["time_ms", "user_id", "video_id"], kind="stable"
    )
    if len(sample) < size:
        raise ValueError(f"Only {len(sample)} plays available")
    sample = sample.iloc[:size]
    interactions = pd.DataFrame(
        {
            "user_id": "user_" + sample.user_id.astype(str),
            "video_id": "video_" + sample.video_id.astype(str),
            "watch_time": (sample.play_time_ms / 1000).round(3),
            "liked": sample.is_like.astype(int),
            "shared": sample.is_forward.astype(int),
            "long_view": sample.long_view.astype(int),
            "timestamp": pd.to_datetime(sample.time_ms, unit="ms", utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
        }
    )

    raw_users = pd.read_csv(raw_dir / "user_features_pure.csv")
    raw_users = raw_users[raw_users.user_id.isin(sample.user_id.unique())]
    users_out = raw_users[["user_id", *USER_COLUMNS]].copy()
    users_out["user_id"] = "user_" + users_out.user_id.astype(str)
    for flag in ["is_lowactive_period", "is_live_streamer", "is_video_author"]:
        users_out[flag] = (users_out[flag] == 1).astype(int)  # anonymised codes such as -124
    for count in ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]:
        users_out[count] = users_out[count].clip(lower=0).astype(int)
    users_out["user_active_degree"] = users_out.user_active_degree.fillna("UNKNOWN").astype(str)

    raw_videos = pd.read_csv(raw_dir / "video_features_basic_pure.csv")
    log_duration = logs.groupby("video_id").duration_ms.median()
    duration_ms = raw_videos.video_duration.fillna(raw_videos.video_id.map(log_duration))
    upload = pd.to_datetime(raw_videos.upload_dt, errors="coerce")
    upload = upload.dt.tz_localize(SOURCE_TIMEZONE).dt.tz_convert("UTC")
    first_seen = pd.to_datetime(logs.groupby("video_id").time_ms.min(), unit="ms", utc=True)
    # A few catalog dates are missing; fall back to the first time the video was logged.
    upload = upload.fillna(raw_videos.video_id.map(first_seen))
    videos_out = pd.DataFrame(
        {
            "video_id": "video_" + raw_videos.video_id.astype(str),
            "creator_id": raw_videos.author_id.astype("int64"),
            "category": raw_videos.tag.map(primary_tag),
            "duration_sec": (duration_ms.fillna(duration_ms.median()) / 1000).clip(lower=0.1),
            "video_type": raw_videos.video_type.fillna("UNKNOWN").astype(str),
            "upload_type": raw_videos.upload_type.fillna("UNKNOWN").astype(str),
            "upload_date": upload.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    videos_out = videos_out.dropna(subset=["upload_date"])
    return users_out, videos_out, interactions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw/KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--mirror", choices=["zenodo", "github"], default="zenodo")
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_dir, output = Path(args.raw_dir), Path(args.output_dir)
    if args.download:
        download(raw_dir, args.mirror)
    users, videos, interactions = build(raw_dir, args.size, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    users.to_csv(output / "users.csv", index=False)
    videos.to_csv(output / "videos.csv", index=False)
    interactions.to_csv(output / "interactions_cleaned.csv", index=False)
    print(
        f"users={len(users):,} videos={len(videos):,} interactions={len(interactions):,} "
        f"from {interactions.timestamp.iloc[0]} to {interactions.timestamp.iloc[-1]}"
    )


if __name__ == "__main__":
    main()
