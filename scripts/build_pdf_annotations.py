#!/usr/bin/env python3
"""Generate EmbedPDF runtime-annotation geometry for local paper PDFs.

Unlike `scripts/annotate_papers.py`, which bakes highlights into the PDF with
text-search coordinates, this script emits a *tracked* JSON manifest of precise
regions that the web reader draws at runtime through the EmbedPDF annotation
plugin (`importAnnotations`). Two region sources are supported:

- ``quote``: a text passage resolved to tight per-line segment rects via a
  ligature/hyphenation-tolerant character-stream match (deterministic).
- ``rect``:  a visually chosen figure/table box, given in PDF points with a
  top-left origin (PyMuPDF space), verified with ``--verify`` overlay images.

Run ``python scripts/build_pdf_annotations.py --verify`` to render overlay
images into ``tmp/annotation_overlays/`` for visual confirmation before writing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "resources" / "papers"
OUT = ROOT / "config" / "paper_annotations.json"
OVERLAYS = ROOT / "tmp" / "annotation_overlays"

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    print("PyMuPDF (fitz) is required: uv add pymupdf", file=sys.stderr)
    sys.exit(1)

# Region fill colors by annotation kind (EmbedPDF runtime rendering).
KIND_COLOR = {
    "conclusion": "#FFCD45",  # warm yellow highlight
    "method": "#8FD3FF",      # blue
    "definition": "#B8F0B0",  # green
    "figure": "#FF9E64",      # orange box
    "table": "#C792EA",       # purple box
}

L = (50.0, 295.0)   # left column x-range (pt) for the 2-col layout
R = (305.0, 560.0)  # right column x-range (pt)

# he2014 (GBDT+LR, Facebook ADKDD'14). Rects are PyMuPDF top-left points,
# visually selected on 2.2x renders (pt = px / 2.2) and overlay-verified.
HE2014 = [
    # --- quick index / headline passages ---
    {"id": "abs-model", "page": 1, "kind": "conclusion", "col": L,
     "quote": "combines decision trees with logistic regression, outperforming either of these methods on its own by over 3%"},
    {"id": "abs-features", "page": 1, "kind": "conclusion", "col": L,
     "quote": "most important thing is to have the right features: those capturing historical information about the user or ad dominate other types of features"},
    {"id": "abs-challenging", "page": 1, "kind": "definition", "col": L,
     "quote": "challenging machine learning task"},
    {"id": "abs-750m", "page": 1, "kind": "definition", "col": L,
     "quote": "over 750 million daily active"},
    # --- method ---
    {"id": "fig1-hybrid", "page": 2, "kind": "figure", "rect": (318, 55, 514, 248)},
    {"id": "eq-ne", "page": 2, "kind": "definition", "rect": (68, 645, 291, 682)},
    {"id": "method-1ofk", "page": 3, "kind": "method", "col": R,
     "quote": "We treat each individual tree as a categorical feature that takes as value the index of the leaf an instance ends up falling in"},
    {"id": "method-onehot", "page": 3, "kind": "method", "col": R,
     "quote": "the overall input to the linear classifier will be the binary vector [0,1,0,1,0]"},
    {"id": "result-3-4pct", "page": 3, "kind": "conclusion", "col": R,
     "quote": "Tree feature transformations help decrease Normalized Entropy by more more than 3.4%"},
    # --- results ---
    {"id": "table1-combination", "page": 4, "kind": "table", "rect": (77, 102, 268, 150)},
    {"id": "fig2-freshness", "page": 4, "kind": "figure", "rect": (59, 170, 302, 320)},
    {"id": "freshness-1pct", "page": 4, "kind": "conclusion", "col": L,
     "quote": "NE can be reduced by approxi-"},
    {"id": "freshness-daily", "page": 4, "kind": "conclusion", "col": L,
     "quote": "by going from training weekly to training daily"},
    {"id": "percoord-sgd", "page": 4, "kind": "conclusion", "col": R,
     "quote": "SGD with per-coordinate learning rate achieves the best prediction accuracy"},
    {"id": "fig4-joiner", "page": 5, "kind": "figure", "rect": (350, 50, 516, 180)},
    {"id": "fig3-lrrate", "page": 5, "kind": "figure", "rect": (61, 159, 302, 311)},
    {"id": "trees-500", "page": 6, "kind": "conclusion", "col": R,
     "quote": "Almost all NE improvement comes from the first 500 trees"},
    {"id": "fig5-trees", "page": 6, "kind": "figure", "rect": (320, 327, 555, 470)},
    {"id": "fig6-importance", "page": 7, "kind": "figure", "rect": (70, 193, 284, 332)},
    {"id": "top10-historical", "page": 7, "kind": "conclusion", "col": R,
     "quote": "The top 10 features ordered by importance are all historical features"},
    {"id": "fig8-historical", "page": 7, "kind": "figure", "rect": (320, 366, 559, 516)},
    {"id": "table4-features", "page": 8, "kind": "table", "rect": (73, 89, 273, 130)},
    {"id": "ctx-4-5-loss", "page": 8, "kind": "conclusion", "col": L,
     "quote": "Without only contextual features, we measure 4.5% loss in prediction accuracy"},
    # --- discussion ---
    {"id": "disc-freshness", "page": 9, "kind": "conclusion", "col": L,
     "quote": "Data freshness matters. It is worth retraining at least daily"},
    {"id": "disc-transform", "page": 9, "kind": "conclusion", "col": L,
     "quote": "Transforming real-valued input features with boosted decision trees signiﬁcantly increases the prediction accuracy of probabilistic linear classifiers"},
]

# grouplens (GroupLens: An Open Architecture for Collaborative Filtering of Netnews,
# CSCW'94). Two-column layout, 622x800 pt. Rects top-left origin.
GROUPLENS = [
    # --- abstract / headline ---
    {"id": "abs-cf-definition", "page": 1, "kind": "definition", "col": L,
     "quote": "Collaborative filters help people make choices based on the opinions of other people"},
    {"id": "abs-heuristic", "page": 1, "kind": "conclusion", "col": L,
     "quote": "people who agreed in the past will probably agree again"},
    {"id": "abs-pseudonyms", "page": 1, "kind": "method", "col": L,
     "quote": "entering ratings under pseudonyms"},
    {"id": "grouplens-architecture", "page": 1, "kind": "method", "col": L,
     "quote": "entire architecture is open"},
    # --- method / core idea ---
    {"id": "cf-idea", "page": 2, "kind": "conclusion", "col": L,
     "quote": "people who agreed in their subjective evaluation of past articles are likely to agree again in the future"},
    {"id": "cf-twostep", "page": 2, "kind": "method", "col": L,
     "quote": "correlates the ratings in order to determine which users ratings are most similar to each other"},
    # --- system / mechanism ---
    {"id": "grouplens-mechanism", "page": 3, "kind": "method", "col": L,
     "quote": "rating servers we have implemented aggregate ratings from several evaluators, based on correlation of their past ratings"},
    {"id": "system-def", "page": 3, "kind": "definition", "col": L,
     "quote": "GroupLens is a distributed system for gathering, disseminating, and using ratings"},
    {"id": "scalability", "page": 9, "kind": "conclusion", "col": L,
     "quote": "We expect prediction quality to increase as the number of users increases, since more data will be available to the prediction algorithm."},
    {"id": "worked-example", "page": 7, "kind": "method", "col": L,
     "quote": "We illustrate one of the correlation and prediction techniques by computing Ken’s predicted score on article 6, the last row of the matrix"},
    # --- figures ---
    {"id": "fig1-netnews", "page": 4, "kind": "figure", "rect": (140, 40, 480, 245)},
    {"id": "fig2-grouplens", "page": 4, "kind": "figure", "rect": (65, 315, 565, 655)},
]

# matrix-factorization (Koren et al., IEEE Computer 2009). Two-column magazine
# layout, 576x792 pt; body columns are ~(48, 285) and (290, 540). Embedded
# figures/sidebars interleave the extracted word stream, so quotes are chosen
# from clean contiguous passages. Rects top-left origin.
MF = [
    # --- overview / conclusion ---
    {"id": "abs-superior", "page": 1, "kind": "conclusion", "col": L,
     "quote": "matrix factorization models are superior to classic nearest-neighbor techniques"},
    {"id": "mf-latent-space", "page": 3, "kind": "definition", "col": L,
     "quote": "Matrix factorization models map both users and items to a joint latent factor space"},
    # --- method ---
    {"id": "sgd", "page": 4, "kind": "method", "col": L,
     "quote": "Stochastic gradient descent"},
    {"id": "als", "page": 4, "kind": "method", "col": L,
     "quote": "Thus, ALS techniques rotate between fixing the qi’s and fixing the pu’s"},
    {"id": "biases", "page": 4, "kind": "method", "col": (290.0, 540.0),
     "quote": "typical collaborative filtering data exhibits large systematic tendencies for some users to give higher ratings than others"},
    {"id": "implicit-feedback", "page": 5, "kind": "method", "col": L,
     "quote": "Recommender systems can use implicit feedback to gain insight into user preferences"},
    {"id": "temporal-dynamics", "page": 5, "kind": "method", "col": L,
     "quote": "account for the temporal effects"},
    # --- results ---
    {"id": "netflix-contest", "page": 6, "kind": "conclusion", "col": L,
     "quote": "the online DVD rental company Netflix announced a contest"},
    {"id": "netflix-rmse", "page": 7, "kind": "conclusion", "col": L,
     "quote": "RMSE = 0.9514"},
    # --- figures ---
    {"id": "fig1-neighborhood", "page": 2, "kind": "figure", "rect": (305, 175, 548, 490)},
    {"id": "fig2-latent-axes", "page": 3, "kind": "figure", "rect": (57, 40, 305, 295)},
    {"id": "fig3-factor-vectors", "page": 6, "kind": "figure", "rect": (360, 85, 548, 345)},
    {"id": "fig4-accuracy", "page": 7, "kind": "figure", "rect": (57, 95, 315, 555)},
]

# fm (Factorization Machines, Steffen Rendle, ICDM 2011). Two-column layout,
# 612x792 pt. Rects top-left origin.
FM = [
    # --- abstract / overview ---
    {"id": "abs-intro", "page": 1, "kind": "definition", "col": L,
     "quote": "In this paper, we introduce Factorization Machines"},
    {"id": "abs-factorized-interactions", "page": 1, "kind": "innovation", "col": L,
     "quote": "model all interactions between variables using factorized parameters"},
    {"id": "abs-linear-time", "page": 1, "kind": "conclusion", "col": R,
     "quote": "can be computed in linear time"},
    {"id": "abs-subsume", "page": 1, "kind": "conclusion", "col": R,
     "quote": "FMs subsume many of the most successful approaches"},
    # --- model ---
    {"id": "fm-equation", "page": 2, "kind": "definition", "col": R,
     "quote": "The model equation for a factorization machine of degree d = 2 is defined as"},
    {"id": "fig1-feature-vector", "page": 2, "kind": "figure", "rect": (50, 40, 305, 170)},
    {"id": "fm-linear-complexity", "page": 3, "kind": "method", "col": L,
     "quote": "This equation has only linear complexity"},
    # --- results / comparison ---
    {"id": "fm-sparse-success", "page": 4, "kind": "conclusion", "col": R,
     "quote": "FMs succeed in estimating 2-way variable interactions in very sparse problems"},
    {"id": "fig2-netflix-error", "page": 4, "kind": "figure", "rect": (360, 40, 560, 215)},
    {"id": "fig3-tag-rec", "page": 6, "kind": "figure", "rect": (55, 50, 270, 275)},
]

# word2vec (Mikolov et al., arXiv 2013 / ICLR 2013). Single-column layout,
# 612x792 pt. Rects top-left origin.
WORD2VEC = [
    # --- abstract / overview ---
    {"id": "abs-architectures", "page": 1, "kind": "definition", "col": (50.0, 560.0),
     "quote": "two novel model architectures for computing continuous vector representations"},
    {"id": "abs-sota", "page": 1, "kind": "conclusion", "col": (50.0, 560.0),
     "quote": "state-of-the-art performance on our test set for measuring syntactic and semantic word similarities"},
    {"id": "abs-fast", "page": 1, "kind": "conclusion", "col": (50.0, 560.0),
     "quote": "less than a day to learn high quality word vectors from a 1.6 billion words data set"},
    # --- model ---
    {"id": "cbow-desc", "page": 4, "kind": "method", "col": (50.0, 560.0),
     "quote": "The first proposed architecture is similar to the feedforward NNLM"},
    {"id": "skipgram-desc", "page": 4, "kind": "method", "col": (50.0, 560.0),
     "quote": "The second architecture is similar to CBOW"},
    {"id": "fig1-architectures", "page": 5, "kind": "figure", "rect": (100, 100, 520, 380)},
    {"id": "cbow-predicts", "page": 5, "kind": "method", "col": (50.0, 560.0),
     "quote": "CBOW architecture predicts the current word based on the context"},
    {"id": "skipgram-predicts", "page": 5, "kind": "method", "col": (50.0, 560.0),
     "quote": "Skip-gram predicts surrounding words given the current word"},
    # --- results ---
    {"id": "skipgram-syntactic", "page": 7, "kind": "conclusion", "col": (50.0, 560.0),
     "quote": "Skip-gram architecture works slightly worse on the syntactic task"},
    {"id": "skipgram-semantic", "page": 7, "kind": "conclusion", "col": (50.0, 560.0),
     "quote": "much better on the semantic part of the test than all the other models"},
    {"id": "table6-distbelief", "page": 9, "kind": "table", "rect": (85, 80, 525, 210)},
    {"id": "training-time", "page": 8, "kind": "conclusion", "col": (50.0, 560.0),
     "quote": "training time for the Skip-gram model was about three days"},
]

# dssm (Huang et al., CIKM 2013). Two-column layout, ~612x792 pt.
DSSM = [
    {"id": "abs-deep", "page": 1, "kind": "definition", "col": L,
     "quote": "we strive to develop a series of new latent semantic models with a deep structure"},
    {"id": "abs-discriminative", "page": 1, "kind": "method", "col": L,
     "quote": "The proposed deep structured semantic models are discriminatively trained by maximizing the conditional likelihood of the clicked documents"},
    {"id": "abs-wordhash", "page": 1, "kind": "method", "col": L,
     "quote": "To make our models applicable to large-scale Web search applications, we also use a technique called word hashing"},
    # Figure/table annotations deliberately box the visual itself.  Highlighting
    # only the caption made the left-hand "structure figure" link appear offset.
    {"id": "fig1-dssm", "page": 3, "kind": "figure", "rect": (30.6, 55.4, 581.4, 277.2)},
    {"id": "eq-relevance", "page": 3, "kind": "definition", "col": L,
     "quote": "The semantic relevance score between a query and a document is then measured as:"},
    {"id": "wordhash-detail", "page": 3, "kind": "method", "col": R,
     "quote": "The word hashing method described here aim to reduce the dimensionality of the bag-of-words term vectors."},
    {"id": "table1-wordhash", "page": 4, "kind": "table", "col": L,
     "quote": "Word hashing token size and collision numbers as a function of the vocabulary size and the type of letter ngrams."},
    {"id": "dssm-results", "page": 6, "kind": "table", "rect": (50.0, 112.0, 300.0, 292.0)},
]

# mind (Li et al., CIKM 2019). Two-column layout, ~612x792 pt.
MIND = [
    {"id": "abs-multi-interest", "page": 1, "kind": "conclusion", "col": L,
     "quote": "we approach this problem from a different view, to represent one user with multiple vectors encoding the different aspects of the user's interests."},
    {"id": "abs-dynamic-routing", "page": 1, "kind": "method", "col": L,
     "quote": "We propose the Multi-Interest Network with Dynamic routing (MIND) for dealing with user's diverse interests in the matching stage."},
    {"id": "fig1-tmall", "page": 1, "kind": "figure", "col": R,
     "quote": "Left: The areas highlighted with dashed rectangle are personalized for billion-scale users at Tmall"},
    {"id": "fig2-mind", "page": 3, "kind": "figure", "rect": (18.4, 63.4, 587.5, 332.6)},
    {"id": "b2i-routing", "page": 5, "kind": "method", "col": L,
     "quote": "Algorithm 1 B2I Dynamic Routing."},
    {"id": "dynamic-interest-number", "page": 5, "kind": "method", "col": L,
     "quote": "calculate adaptive number of interest capsules K by (9)"},
    {"id": "label-aware-attention", "page": 5, "kind": "method", "col": L,
     "quote": "label-aware attention, the label is the query and the interest capsules are both keys and values"},
    {"id": "mind-offline-results", "page": 7, "kind": "table", "rect": (50.0, 80.0, 565.0, 232.0)},
    {"id": "mind-online-results", "page": 7, "kind": "figure", "rect": (310.0, 245.0, 563.0, 442.0)},
]

# sasrec (Kang & McAuley, IEEE 2018). Two-column layout, ~612x792 pt.
SASREC = [
    {"id": "abs-sequential-dynamics", "page": 1, "kind": "definition", "col": L,
     "quote": "Sequential dynamics are a key feature of many modern recommender systems, which seek to capture the 'context' of users' activities on the basis of actions they have performed recently."},
    {"id": "abs-self-attention", "page": 1, "kind": "method", "col": L,
     "quote": "by proposing a self-attention based sequential model (SASRec)"},
    {"id": "fig1-sasrec", "page": 1, "kind": "figure", "rect": (312.1, 142.6, 605.9, 332.6)},
    {"id": "notation-table", "page": 3, "kind": "definition", "col": (50.0, 560.0),
     "quote": "Table I: Notation."},
    {"id": "embedding-layer", "page": 3, "kind": "method", "col": L,
     "quote": "We create an item embedding matrix"},
    {"id": "causal-mask", "page": 3, "kind": "method", "col": R,
     "quote": "Hence, we modify the attention by forbidding Qi Kj (j > i)."},
    {"id": "self-attention-block", "page": 3, "kind": "method", "col": R,
     "quote": "The scaled dot-product attention [3] is defined as:"},
    {"id": "sasrec-results", "page": 7, "kind": "table", "rect": (48.0, 45.0, 565.0, 195.0)},
]


# deepfm (Guo et al., IJCAI 2017). Two-column layout, ~612x792 pt.
DEEPFM = [
    {"id": "fig1-deepfm", "page": 1, "kind": "figure", "col": R,
     "quote": "Wide & deep architecture of DeepFM. The wide and deep component share the same input raw feature vector, which enables DeepFM to learn low- and high-order feature interactions simultaneously from the input raw features."},
    {"id": "abs-intro", "page": 1, "kind": "definition", "col": L,
     "quote": "DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture."},
    {"id": "abs-shared-input", "page": 1, "kind": "innovation", "col": L,
     "quote": "DeepFM has a shared input to its \"wide\" and \"deep\" parts, with no need of feature engineering besides raw features."},
    {"id": "model-components", "page": 2, "kind": "method", "col": L,
     "quote": "It models low-order feature interactions like FM and models high-order feature interactions like DNN. Unlike the wide & deep model [Cheng et al., 2016], DeepFM can be trained end-to-end without any feature engineering."},
    {"id": "shared-embedding", "page": 2, "kind": "method", "col": L,
     "quote": "share the same input and also the embedding vector."},
    {"id": "fm-component", "page": 2, "kind": "method", "col": L,
     "quote": "inner product of latent vectors between features and show very promising results"},
    {"id": "deepfm-results", "page": 5, "kind": "conclusion", "col": L,
     "quote": "As the best model, DeepFM outperforms LR by 0.86% and 4.18% in terms of AUC (1.15% and 5.60% in terms of Logloss) on Company* and Criteo datasets."},
    {"id": "table2-deepfm", "page": 5, "kind": "table", "col": R,
     "quote": "Table 2: Performance on CTR prediction."},
    {"id": "deepfm-shared-win", "page": 5, "kind": "conclusion", "col": R,
     "quote": "Compared to these two models, DeepFM achieves more than 0.48% and 0.33% in terms of AUC (0.61% and 0.66% in terms of Logloss) on Company* and Criteo datasets."},
]

# din (Zhou et al., KDD 2018). Two-column layout, ~612x792 pt.
DIN = [
    {"id": "fig2-din", "page": 4, "kind": "figure", "col": L,
     "quote": "Figure 2: Network Architecture"},
    {"id": "abs-fixed-vector", "page": 1, "kind": "conclusion", "col": L,
     "quote": "user features are compressed into a fixed-length representation vector, in regardless of what candidate ads are."},
    {"id": "abs-din-proposed", "page": 1, "kind": "innovation", "col": L,
     "quote": "we propose a novel model: Deep Interest Network (DIN) which tackles this challenge by designing a local activation unit to adaptively learn the representation of user interests from historical behaviors with respect to a certain ad."},
    {"id": "local-activation", "page": 4, "kind": "method", "col": R,
     "quote": "This representation vector varies over different ads."},
    {"id": "local-activation-formula", "page": 4, "kind": "definition", "col": R,
     "quote": "vU (A) = f (vA, e1, e2, .., eH ) = a(ej, vA)ej = wjej, (3)"},
    {"id": "dice-activation", "page": 6, "kind": "method", "col": L,
     "quote": "Dice can be viewed as a generalization of PReLu. The key idea of Dice is to adaptively adjust the rectified point according to distribution of input data, whose value is set to be the mean of input."},
    {"id": "din-metrics", "page": 6, "kind": "method", "col": R,
     "quote": "In CTR prediction field, AUC is a widely used metric"},
    {"id": "din-results", "page": 6, "kind": "conclusion", "col": L,
     "quote": "proposed approach which outperforms state-of-the-art methods on the CTR prediction task"},
    {"id": "din-online-ab", "page": 8, "kind": "conclusion", "col": L,
     "quote": "DIN trained with the proposed regularizer and activation function contributes up to 10.0% CTR and 3.8% RPM(Revenue Per Mille) promotion"},
]

# dien (Zhou et al., AAAI 2019). Two-column layout, ~612x792 pt.
DIEN = [
    {"id": "fig1-dien", "page": 4, "kind": "figure", "col": (50.0, 560.0),
     "quote": "Figure 1: The structure of DIEN."},
    {"id": "abs-dien", "page": 1, "kind": "definition", "col": L,
     "quote": "we propose a novel model, named Deep Interest Evolution Network (DIEN), for CTR prediction. Specifically, we design interest extractor layer to capture temporal interests from history behavior sequence."},
    {"id": "abs-aux-loss", "page": 1, "kind": "method", "col": L,
     "quote": "At this layer, we introduce an auxiliary loss to supervise interest extracting at each step."},
    {"id": "aux-loss-formula", "page": 4, "kind": "definition", "col": L,
     "quote": "Auxiliary loss can be formulated as:"},
    {"id": "augru-intro", "page": 1, "kind": "method", "col": R,
     "quote": "we design GRU with attentional update gate (AUGRU)"},
    {"id": "interest-evolving", "page": 4, "kind": "method", "col": R,
     "quote": "we combine the local activation ability of attention mechanism and sequential learning ability from GRU to model interest evolving. The local activation during each step of GRU can intensify relative interest's effect, and weaken the disturbance from interest drifting"},
    {"id": "dien-online", "page": 7, "kind": "table", "col": L,
     "quote": "Table 5: Results from Online A/B testing"},
    {"id": "dien-gains", "page": 7, "kind": "conclusion", "col": R,
     "quote": "mille (eCPM) by 17.1%."},
    {"id": "dien-serving", "page": 7, "kind": "method", "col": R,
     "quote": "latency of DIEN serving can be reduced from 38.2 ms to 6.6 ms and the QPS (Query Per Second) capacity of each worker can be improved to 360."},
]


# mmoe (Ma et al., KDD 2018). Two-column layout, ~612x792 pt.
MMOE = [
    {"id": "fig1-mmoe", "page": 2, "kind": "figure", "col": L,
     "quote": "Figure 1: (a) Shared-Bottom model. (b)"},
    {"id": "abs-mmoe", "page": 1, "kind": "definition", "col": L,
     "quote": "Multi-gate Mixture-of-Experts (MMoE), which explicitly learns to model task relationships from data."},
    {"id": "mmoe-idea", "page": 2, "kind": "method", "col": L,
     "quote": "Instead of having one bottom network shared by all tasks, our model, shown in Figure 1 (c), has a group of bottom networks, each of which is called an expert."},
    {"id": "gating-network", "page": 2, "kind": "method", "col": L,
     "quote": "We then introduce a gating network for each task. The gating networks take the input features and output softmax gates assembling the experts with different weights, allowing different tasks to utilize experts differently."},
    {"id": "mmoe-formula", "page": 5, "kind": "definition", "col": L,
     "quote": "where f k (x) ="},
    {"id": "mmoe-trainability", "page": 6, "kind": "conclusion", "col": L,
     "quote": "the MoE models have better trainability than the Shared-Bottom model"},
    {"id": "mmoe-results", "page": 8, "kind": "conclusion", "col": L,
     "quote": "MMoE outperforms other multi-task models in all means in group 2, where the task relatedness is even smaller than group 1."},
    {"id": "mmoe-google", "page": 8, "kind": "conclusion", "col": R,
     "quote": "MMoE outperforms other models in terms of both metrics."},
    {"id": "fig4-mmoe", "page": 5, "kind": "figure", "rect": (50, 45, 560, 236)},
    {"id": "table3-mmoe", "page": 9, "kind": "table", "rect": (50, 80, 560, 180)},
    {"id": "table4-mmoe", "page": 9, "kind": "table", "rect": (305, 190, 560, 322)},
]

# ple (Tang et al., RecSys 2020). Two-column layout, ~612x792 pt.
PLE = [
    {"id": "fig4-ple", "page": 5, "kind": "figure", "col": L,
     "quote": "Figure 4: Customized Gate Control (CGC) Model"},
    {"id": "abs-ple", "page": 1, "kind": "definition", "col": L,
     "quote": "PLE separates shared components and task-specific components explicitly"},
    {"id": "seesaw", "page": 1, "kind": "conclusion", "col": L,
     "quote": "seesaw phenomenon that performance of one task is often improved by hurting the performance of some other tasks."},
    {"id": "cgc-gate", "page": 5, "kind": "method", "col": R,
     "quote": "In CGC, shared experts and task-specific experts are combined through a gating network for selective fusion."},
    {"id": "cgc-formula", "page": 5, "kind": "definition", "col": R,
     "quote": "Finally, the prediction of task k is:"},
    {"id": "ple-progressive", "page": 6, "kind": "method", "col": L,
     "quote": "PLE extracts and combines deeper semantic representations for each task to improve generalization"},
    {"id": "ple-results", "page": 8, "kind": "conclusion", "col": L,
     "quote": "PLE converges with similar pace and achieves significant improvement over the above models with the best VCR MSE and one of the best VTR AUCs."},
    {"id": "ple-online", "page": 8, "kind": "table", "col": L,
     "quote": "Table 3: Improvement over Single-task Model on Online A/B Test"},
    {"id": "fig3-ple", "page": 4, "kind": "figure", "rect": (345, 460, 525, 576)},
    {"id": "table1-ple", "page": 7, "kind": "table", "rect": (305, 390, 560, 550)},
    {"id": "fig8-ple", "page": 9, "kind": "figure", "rect": (48, 512, 302, 709)},
    {"id": "table5-ple", "page": 10, "kind": "table", "rect": (50, 80, 562, 172)},
]


# tiger (Mehta et al., NeurIPS 2023). Two-column layout, ~612x792 pt.
TIGER = [
    {"id": "fig1-tiger", "page": 1, "kind": "figure", "col": (50.0, 560.0),
     "quote": "Figure 1: Overview of"},
    {"id": "tiger-proposed", "page": 1, "kind": "innovation", "col": (50.0, 560.0),
     "quote": "the retrieval model autoregressively decodes the identifiers of the target candidates."},
    {"id": "tiger-semantic-id", "page": 1, "kind": "method", "col": (50.0, 560.0),
     "quote": "Semantic ID for each item."},
    {"id": "tiger-encoder-decoder", "page": 3, "kind": "method", "col": (50.0, 560.0),
     "quote": "based encoder-decoder setup for building"},
    {"id": "tiger-results", "page": 1, "kind": "conclusion", "col": (50.0, 560.0),
     "quote": "outperform the current SOTA models on various datasets"},
    {"id": "tiger-rqvae", "page": 4, "kind": "method", "col": (50.0, 560.0),
     "quote": "is a multi-level vector quantizer that applies quantization on residuals"},
    {"id": "tiger-collision", "page": 5, "kind": "method", "col": (50.0, 560.0),
     "quote": "we append an extra token at the end of the ordered semantic codes to make them unique"},
    {"id": "tiger-table1", "page": 7, "kind": "table", "col": (50.0, 560.0),
     "quote": "Table 1: Performance comparison on sequential recommendation."},
    {"id": "tiger-table2", "page": 8, "kind": "table", "col": (50.0, 560.0),
     "quote": "Table 2: Ablation study for different ID generation techniques"},
]

# onerec (Deng et al., arXiv 2025). Two-column layout, ~612x792 pt.
ONEREC = [
    {"id": "fig2-onerec", "page": 3, "kind": "figure", "col": (50.0, 560.0),
     "quote": "Figure 2: The overall framework of OneRec"},
    {"id": "onerec-proposed", "page": 1, "kind": "innovation", "col": L,
     "quote": "we propose OneRec, which replaces the cascaded learning framework with a unified generative model."},
    {"id": "onerec-encoder-decoder", "page": 1, "kind": "method", "col": L,
     "quote": "an encoder-decoder structure, which encodes the user's historical behavior sequences and gradually decodes the videos that the user may be interested in."},
    {"id": "onerec-moe", "page": 1, "kind": "method", "col": L,
     "quote": "We adopt sparse Mixture-of-Experts (MoE) to scale model capacity without proportionally increasing computational FLOPs."},
    {"id": "onerec-session", "page": 1, "kind": "method", "col": L,
     "quote": "we propose a session-wise generation, which is more elegant and contextually coherent than point-by-point generation"},
    {"id": "onerec-ipa", "page": 1, "kind": "method", "col": L,
     "quote": "an Iterative Preference Alignment module combined with Direct Preference Optimization (DPO) to enhance the quality of the generated results."},
    {"id": "onerec-results", "page": 1, "kind": "conclusion", "col": R,
     "quote": "16 increase in watch time"},
    {"id": "onerec-balanced-kmeans", "page": 3, "kind": "method", "col": R,
     "quote": "We apply a multi-level balanced quantitative mechanism to transform the"},
    {"id": "onerec-table1", "page": 6, "kind": "table", "col": L,
     "quote": "Table 1: Offline performance of our proposed OneRec"},
    {"id": "onerec-deploy", "page": 6, "kind": "method", "col": L,
     "quote": "during inference only 13% of the parameters are activated"},
    {"id": "onerec-table2", "page": 8, "kind": "table", "col": R,
     "quote": "Table 2: The absolute improvement of OneRec compared"},
]

# hstu (Zhai et al., ICML 2024). Two-column layout, ~612x792 pt.
HSTU = [
    {"id": "fig3-hstu", "page": 4, "kind": "figure", "col": R,
     "quote": "Figure 3. Comparison of key model components: DLRMs vs GRs."},
    {"id": "hstu-proposed", "page": 1, "kind": "definition", "col": L,
     "quote": "propose a new architecture, HSTU, designed for high cardinality, non-stationary streaming recommendation data."},
    {"id": "hstu-equations", "page": 4, "kind": "definition", "col": L,
     "quote": "Each layer contains three sub-layers: Pointwise Projection (Equation 1), Spatial Aggregation (Equation 2), and Pointwise Transformation (Equation 3):"},
    {"id": "hstu-pointwise", "page": 4, "kind": "method", "col": R,
     "quote": "HSTU adopts a new pointwise aggregated (normalized) attention mechanism"},
    {"id": "hstu-results", "page": 7, "kind": "table", "col": L,
     "quote": "Table 4. Evaluations of methods on public"},
    {"id": "hstu-online", "page": 7, "kind": "table", "col": R,
     "quote": "Table 7. Offline/Online Comparison of Ranking Models."},
    {"id": "hstu-ablation", "page": 7, "kind": "table", "col": L,
     "quote": "Table 5. Evaluation of HSTU, ablated HSTU, and Transformers"},
    {"id": "hstu-efficiency", "page": 7, "kind": "conclusion", "col": L,
     "quote": "HSTU is up to 15.2x and 5.6x more efficient"},
    {"id": "hstu-scaling", "page": 8, "kind": "conclusion", "col": R,
     "quote": "GRs leading to 1.5 trillion parameter models, whereas DLRMs performance saturate at about 200 billion parameters"},
]

SPECS = {"he2014": HE2014, "grouplens": GROUPLENS, "matrix-factorization": MF, "fm": FM, "word2vec": WORD2VEC, "dssm": DSSM, "mind": MIND, "sasrec": SASREC, "deepfm": DEEPFM, "din": DIN, "dien": DIEN, "mmoe": MMOE, "ple": PLE, "tiger": TIGER, "onerec": ONEREC, "hstu": HSTU}

# Some SPECS keys name the PDF file while the runtime paper id (resources.json,
# evidence items, /papers/{id} routes) uses a shorter alias. Build/verify read
# the PDF by SPECS key; the JSON is written under the runtime id.
OUTPUT_ALIASES = {"matrix-factorization": "mf"}


def _key(t: str) -> str:
    t = (t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
         .replace("ﬃ", "ffi").replace("ﬄ", "ffl"))
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _norm(s: str) -> str:
    s = (s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", s).strip().lower()


def _words_in(doc, page_no: int, x_min: float, x_max: float):
    ws = doc[page_no - 1].get_text("words")
    ws = [w for w in ws if w[2] >= x_min and w[0] <= x_max]
    ws.sort(key=lambda w: (round(w[1] / 3), w[0]))
    return ws


def resolve_quote(doc, page_no: int, col, quote: str):
    """Resolve a passage to tight per-line segment rects (PyMuPDF top-left pt).

    Matches the concatenated alphanumeric character stream so hyphenation and
    ligature splits inside words do not break the match."""
    ws = _words_in(doc, page_no, col[0], col[1])
    if not ws:
        return None
    joined = ""
    offs = []
    for w in ws:
        k = _key(w[4])
        offs.append((len(joined), len(joined) + len(k)))
        joined += k
    qjoined = "".join(_key(t) for t in _norm(quote).split(" "))
    if not qjoined:
        return None
    idx = joined.find(qjoined)
    if idx < 0:
        return None
    qend = idx + len(qjoined)
    a = next(i for i, (s, e) in enumerate(offs) if s <= idx < e)
    b = next(i for i, (s, e) in enumerate(offs) if s < qend <= e)
    sel = ws[a:b + 1]
    byline = defaultdict(list)
    for w in sel:
        byline[round(w[1] / 3)].append(w)
    rects = []
    for key in sorted(byline):
        g = byline[key]
        rects.append((min(w[0] for w in g), min(w[1] for w in g),
                      max(w[2] for w in g), max(w[3] for w in g)))
    return rects


def _to_ep(rect, page_h: float):
    """PyMuPDF rect -> EmbedPDF {origin,size}.

    EmbedPDF's annotation model uses a *top-left* origin with y increasing
    downward (the engine flips to PDF bottom-left only when writing /Rect back
    to pdfium). PyMuPDF is also top-left, so coordinates pass through unchanged.
    ``page_h`` is kept in the signature for callers that already computed it."""
    x0, y0, x1, y1 = rect
    return {
        "origin": {"x": round(x0, 1), "y": round(y0, 1)},
        "size": {"width": round(x1 - x0, 1), "height": round(y1 - y0, 1)},
    }


def build(paper_id: str, spec: list[dict]) -> dict:
    pdf = PAPERS / f"{paper_id}.pdf"
    if not pdf.exists():
        return {"paper": paper_id, "status": "missing"}
    doc = fitz.open(pdf)
    page_h = doc[0].rect.height
    annotations = []
    unresolved = []
    for a in spec:
        if "rect" in a:
            rects = [tuple(a["rect"])]
        else:
            rects = resolve_quote(doc, a["page"], a["col"], a["quote"])
            if not rects:
                unresolved.append(a["id"])
                continue
        box = a.get("rect")
        if box is None:
            x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
            x1 = max(r[2] for r in rects); y1 = max(r[3] for r in rects)
            box = (x0, y0, x1, y1)
        annotations.append({
            "id": a["id"],
            "page": a["page"],            # 1-based, matches evidence anchors
            "kind": a["kind"],
            "color": KIND_COLOR.get(a["kind"], "#FFCD45"),
            "is_text": "quote" in a,       # highlight (segmentRects) vs box (square)
            "rect": _to_ep(box, page_h),
            "segmentRects": [_to_ep(r, page_h) for r in rects] if "quote" in a else None,
        })
    doc.close()
    return {"paper": paper_id, "status": "ok", "page_height": page_h,
            "annotations": annotations, "unresolved": unresolved}


def verify(paper_id: str, annotations: list[dict]) -> None:
    pdf = PAPERS / f"{paper_id}.pdf"
    doc = fitz.open(pdf)
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    for a in annotations:
        page = doc[a["page"] - 1]
        shapes = a["segmentRects"] or [a["rect"]]
        for s in shapes:
            # top-left origin (EmbedPDF annotation model == PyMuPDF space)
            x0 = s["origin"]["x"]; y0 = s["origin"]["y"]
            x1 = x0 + s["size"]["width"]; y1 = y0 + s["size"]["height"]
            r = fitz.Rect(x0, y0, x1, y1)
            page.draw_rect(r, color=fitz.utils.getColor("red"), width=1.2)
    for pno in range(doc.page_count):
        pix = doc[pno].get_pixmap(matrix=fitz.Matrix(1.6, 1.6))
        pix.save(OVERLAYS / f"{paper_id}_p{pno + 1:02d}.png")
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="append", help="paper id (repeatable); default all known")
    parser.add_argument("--verify", action="store_true", help="render overlay images into tmp/annotation_overlays/")
    args = parser.parse_args()
    result = {}
    for pid in (args.paper or list(SPECS)):
        built = build(pid, SPECS[pid])
        if built["status"] != "ok":
            print(built)
            continue
        out_id = OUTPUT_ALIASES.get(pid, pid)
        result[out_id] = {"page_height": built["page_height"], "annotations": built["annotations"]}
        print(f"{out_id}: {len(built['annotations'])} annotations, unresolved={built['unresolved']}")
        if args.verify:
            verify(pid, built["annotations"])
            print(f"  overlays -> {OVERLAYS}")
    if not args.verify:
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
