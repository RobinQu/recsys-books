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

## OpenOneRec RecIF-Bench access boundary

The official `OpenOneRec/OpenOneRec-RecIF` Hugging Face repository is gated as
of this build. The smoke notebook therefore uses observed KuaiRand interactions
and video taxonomy through an explicitly named compatibility adapter; it does
not claim to reproduce the RecIF-Bench score. The full profile requires the
reader to accept the official license and authenticate before downloading the
RecIF parquet and semantic-ID mappings.
