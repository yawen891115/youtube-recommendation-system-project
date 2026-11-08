# Data exploration

The supplied dataset contains 10,000 users, 10,000 videos and 100,000 interactions.

Like rate: 9.30%. Share rate: 2.32%. Mean watch duration: 33.51 seconds.

Observation window: 2024-01-01T00:05:18+00:00 to 2024-11-30T23:59:26+00:00.

The three raw tables contain no missing cells or duplicate rows and all interaction foreign keys resolve. Rewatch observations are retained; completion rate is capped at one. There are 9,469 interactions dated before the catalog upload date. Raw rows are preserved; negative video age is clipped to zero. This inconsistency limits realism. Catalog metadata is treated as a static snapshot; its historical availability cannot be verified. The supplied materials do not establish that this is live YouTube traffic.

## Split distributions

- train: 64,000 rows, 9.45% positive.
- validation: 16,000 rows, 9.18% positive.
- test: 20,000 rows, 8.95% positive.
