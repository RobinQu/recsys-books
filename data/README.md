# Dataset provenance

All executable notebooks use the bundled **GroupLens MovieLens latest-small**
release. The original files are stored in `data/ml-latest-small/` and were
downloaded from the official GroupLens distribution:

- Source: <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
- Dataset documentation: <https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html>
- Release timestamp recorded by GroupLens: 2018-09-26
- Original scale: 100,836 ratings, 610 users and 9,742 movies
- `ratings.csv` SHA-256: `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646`
- `movies.csv` SHA-256: `5a5f32dd9bb3797b8e728a1b98958789d2b13f294a69fdfbc5727f8a9611aa07`

The smoke profile deterministically selects active users and frequent movies
so every notebook can run on CPU. It never creates synthetic interactions,
ratings, labels or behavior sequences. Model seeds only initialize parameters
and control repeatability.

Task-specific fields are derived transparently from the original rows:

- positive preference: `rating >= 4.0`
- strong preference: `rating >= 4.5`
- observed negative preference: an original low-rating event
- sequence order: original Unix timestamp
- hour and weekday: converted from the original timestamp
- genre and release decade: parsed from the original movie metadata

MovieLens is an explicit-rating dataset, not an impression/click log. Therefore
binary tasks in this tutorial are preference-classification proxies and their
AUC or LogLoss values must not be described as production CTR/CVR results.
The dataset's usage and redistribution terms remain those stated in the bundled
`ml-latest-small/README.txt`.
