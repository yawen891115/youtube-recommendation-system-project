# Source version audit

The source archives were compared by extracted file contents, SHA-256, archive timestamps and implemented behavior. Archive numbering alone was not treated as a version guarantee. The consolidated project is a completed revision of the latest substantive supplied code, not an unmodified copy of a ZIP.

## Archive comparison

| Supplied archive | Finding | Decision |
| --- | --- | --- |
| 8.6youtube-recommendation-system-project(1).zip | Latest core implementation, August 6, 2026 | Selected base |
| youtube-recommendation-system-project 5(1).zip | Core files match the 8.6 archive byte for byte | Same latest base |
| youtube-recommendation-system-project 4(1).zip | Same latest core; also contains an old numbered training script with a misleading August 20 timestamp | Use current named modules, omit superseded script |
| youtube-recommendation-system-project 2(1).zip | Intermediate August 6 training/UI revision | Superseded; English assignment recovered |
| old youtube(1).zip | Earlier numbered modules and earlier August 6 API/UI | Superseded |
| youtube-recommendation-system-project 3(1).zip | Earlier July 30 implementation | Superseded |
| youtube-recommendation-system-project(1).zip | Earlier July 30 implementation with original English assignment | Assignment preserved unchanged |
| Machine Learning System Design Interview(2)(1).zip | General reference-book materials, not a newer implementation or additional project dataset | Not republished as project code |
| YouTube Shorts 视频推荐系统_中文翻译(1).md | Chinese translation of the assignment | English original selected for this repository |

## Latest substantive core files

These hashes identify the supplied base files before the completion work. The current modules contain the documented fixes and additions.

| File | Archive timestamp | Source SHA-256 |
| --- | --- | --- |
| streamlit_app.py | 2026-08-06 20:28:36 | `174a0000b27a73d20b236901208f5b8c90e4ae80988d5c94eaa9aab1570fec8e` |
| src/feature_engineering.py | 2026-08-06 20:26:58 | `68a8608fbc878a38356a4a67912a218f5d0c79334cd93a1564df30e5fbc0070c` |
| src/data_validation.py | 2026-08-06 20:25:18 | `4027b09ef5d4ccb82c6d962ba432ae849bda8d667c36160f16e2fe7c64dbcbce` |
| src/api.py | 2026-08-06 20:25:08 | `6c79fc1f4ed12cfdef17f53ebb18a7f955e7b3f1492881cc709cce0268c6640b` |
| src/train.py | 2026-08-06 20:25:14 | `80bc17159d822c69e24aa6b60be0e202f295a91f2396becf865abe7bb5331eb4` |

The August 20 timestamp on `src/03_train.py` in archive 4 does not indicate a newer implementation: its content hash matches the older July 30 script. The August 6 named modules are the later substantive implementation.

## Data and document selection

All seven project archives contain identical raw table checksums. The original generated feature CSV also has identical content across those versions despite different file timestamps. It was regenerated because the current pipeline fixes history leakage and uses the required chronological split. Old serialized models were not executed; fresh artifacts were trained from the included raw data.

The English assignment was recovered from the early project archive and copied unchanged to `assignment-original.en.md`. Its SHA-256 is `cd869b0982aaca222a662da7869a891e667f0220a839ba9f9ab3364fdaf90a9f`.

## Completion work

The consolidated revision fixes future/current-label history leakage, aligns training and serving features, adds cold-start handling and strict validation, implements validation-only early stopping and tuning, includes an additional pre-watch ranker, and completes Redis caching, rate limits, Prometheus monitoring, the English UI, deployment files, tests and reproducible reports.

Old `.git` directories, environment caches, duplicate source versions and stale generated artifacts are excluded from the working tree. The destination repository retains its existing Git history through a normal descendant commit. Machine-readable source evidence is in `version-manifest.json`.
