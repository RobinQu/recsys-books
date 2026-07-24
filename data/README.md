# Dataset provenance and algorithm mapping

The tutorial uses different real datasets for different recommendation tasks.
Committed files are deterministic CPU slices; `scripts/prepare_dataset_slices.py`
reproduces the selection from the official archives. Model seeds never create
interactions, exposures, labels, ratings or sequences.

| Chapters | Dataset | Why it fits |
|---|---|---|
| 3.0–3.1 | MovieLens latest-small | Small explicit-rating matrix for CF, MF and feature-cross teaching |
| 3.2 | Amazon Reviews 2023 / Video Games 5-core | Modern e-commerce users, items, ratings and fine-grained timestamps for dual-tower and multi-interest retrieval |
| 3.3–3.4 | KuaiRand-Pure | Real feed impressions with click, long-view, like and other feedback for ranking and multitask learning |
| 3.5 | Amazon Reviews 2023 / Video Games 5-core | Real chronological product sequences through September 2023 for SASRec |
| 4.2–4.3 | KuaiRand-Pure | Short-video taxonomy and non-stationary feed sequences for constrained-generation teaching and HSTU |

## MovieLens latest-small

- Source: <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
- Official scale: 100,836 ratings, 610 users and 9,742 movies
- `ratings.csv` SHA-256: `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646`
- Use: only the classic teaching chapters
- Terms: see `ml-latest-small/README.txt`

## Amazon Reviews 2023 / Video Games / 5-core

- Project: <https://amazon-reviews-2023.github.io/>
- Source: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz>
- Source SHA-256: `a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c`
- Official 5-core scale: 94.8K users, 25.6K items and 814.6K ratings
- Committed slice: the 500 most active users, ties by user ID, retaining all 39,025 observed rows
- Fields: `user_id`, `parent_asin`, `rating`, millisecond `timestamp`
- Positive preference: observed `rating >= 4.0`; low-rating negatives are observed rows
- Exact machine-readable record: `amazon-reviews-2023-video-games/provenance.json`

## KuaiRand-Pure

- Project and schema: <https://github.com/chongminggao/KuaiRand>
- Source: <https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz>
- Source SHA-256: `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`
- License: CC-BY-SA-4.0; see `kuairand-pure/LICENSE`
- Official Pure scale: 1.44M standard-policy and 1.19M random-policy interactions over a 7.5K-video pool
- Committed slice: the 256 most active standard-feed users, retaining all their standard and random-policy rows
- Native targets used without threshold fabrication: `is_click`, `long_view`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`
- Exact machine-readable record: `kuairand-pure/provenance.json`

## Full-profile external datasets

These live under ignored `resources/datasets/` and are fetched by
`python scripts/init_resources.py --include-optional --kind datasets --id <id>`:

- `movielens-latest-full`: official MovieLens latest zip (~33M ratings,
  GroupLens regenerates it periodically, so loaders assert a 33M floor and
  record actual counts in provenance). Used by every chapter 4 classic
  notebook in full profile.
- `movielens-1m-full` / `movielens-20m-full`: stable GroupLens releases for
  SASRec (5.4) and HSTU (8.3) paper protocols.
- `amazon-books-5core` / `amazon-electronics-5core` /
  `amazon-books-2014-ratings`: paper-aligned retrieval/ranking corpora
  (5.2, 5.3, 6.3, 6.4).
- `criteo-x1-full`: reczoo Criteo_x1 (HF), official 7:2:1 split
  33,003,326 / 8,250,124 / 4,587,167 rows, sha256-pinned. Full profile for
  GBDT+LR (4.5) and DeepFM (6.2).
- `census-income-kdd`: MMoE/PLE (7.2, 7.3) official train/test.
- `openonerec-recif`: gated, see the next section.

## OpenOneRec RecIF-Bench access boundary

The official `OpenOneRec/OpenOneRec-RecIF` Hugging Face repository is gated:
each reader must accept the official terms and authenticate with their own
`HF_TOKEN` before downloading. An authorized local copy (Apache License 2.0,
see `resources/datasets/recif-bench/LICENSE`) now lives at
`resources/datasets/recif-bench/` (~8.4 GB, snapshot 8f7cf2ee, fetched
2026-07-24): `onerec_bench_release.parquet` (162,074 users with per-domain
hist/target feedback), `benchmark_data/` (8 task test sets plus
`sid2iid.json`/`sid2pid.json`), `video_ad_pid2sid.parquet` (15.9M pid→sid),
`product_pid2sid.parquet` (2.07M pid→sid) and `pid2caption.parquet` (12.7M
dense captions). The smoke notebook still uses observed KuaiRand interactions
and video taxonomy through an explicitly named compatibility adapter; it does
not claim to reproduce the RecIF-Bench score. Full-profile RecIF runs read
only this local copy and never redistribute it.
