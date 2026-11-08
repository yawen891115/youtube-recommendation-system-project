# Dataset

All three raw CSV files are included in full and are byte-identical to the supplied project archives. There is no external data download or credentials requirement. The data source is the user-supplied project package; the materials do not establish that it contains live YouTube traffic.

| File | Rows | Fields |
| --- | ---: | --- |
| users.csv | 10,000 | user_id, age, gender, preferred_category, join_date |
| videos.csv | 10,000 | video_id, creator_id, category, duration_sec, view_count, upload_date |
| interactions_cleaned.csv | 100,000 | user_id, video_id, watch_time, liked, shared, timestamp |

Identifiers are pseudonymous strings. Dates are normalized to UTC by the loader; the original CSV bytes are preserved. The prediction label is `liked`; `shared` is used only in prior-history aggregates. `view_count` is a catalog snapshot and is excluded from historical features to avoid using an unknown-time aggregate.

Run `python -m src.data_validation` from the project root for integrity checks and distributions. Run `sha256sum -c checksums.sha256` from this directory to verify original file bytes. See `../reports/data-exploration.md` for timestamp anomalies and split statistics.

The processed `.csv.xz` and `.parquet` each contain all 100,000 rows: IDs, timestamp, labels, split and 167 derived features. They are rebuilt from the raw tables by the training command. Their labels and split columns must not be used as model input. The original English assignment contains the detailed table definitions.
