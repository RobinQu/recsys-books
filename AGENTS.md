# RecSys Atlas engineering guide

## Product structure

Keep exactly two information levels: the home page and an algorithm/notebook detail page. Do not add category landing pages. The global top bar and left navigation are shared templates; a detail page only changes active state. The home page ends with one Appendix chapter containing paper, dataset, Notebook, and math-knowledge-map (A.4) subchapters; do not restore a framework-selection section. Every detail page has exactly four core tabs: **论文导读**, **实验预览**, **可交互执行**, and **代码实现**.

### Detail-page tab layout and behavior

The four tabs live in a single tab bar under the page header (`app/templates/notebook_shell.html`).

1. **论文导读 (Paper Evidence)**
   - Shown only when `show_paper_guide` is true. `app/content.py:notebook_has_paper_guide` returns `False` for the foundations chapter (3), every chapter opening/导读 page, and every summary page (4_7_classic_summary / 5_5_retrieval_summary / 6_5_ranking_summary / 7_4_multitask_summary / 8_4_generative_summary); only algorithm detail pages keep it. Summary pages do their cross-paper comparison (论文关联关系, 实验数据审计, 未来发展) inside the summary notebook instead.
   - Default active tab when available; otherwise the default falls back to 实验预览.
   - Split-pane layout on desktop: left prose, right embedded PDF viewer.
   - **Left pane (prose):**
     - Heading with `paper_guide.intro`.
     - `quick_index`: clickable cards labeled by `kind` (definition / method / conclusion / figure / table) that jump the right PDF to the matching page/annotation.
     - `guide`: an eight-field schema (`background`, `problem`, `innovation`, `method`, `findings`, `contribution`, `limitations`, `one_liner`). Prose may contain inline `[[term|page|evidence_id]]` links parsed by `app/evidence.py:_parse_prose_links`; each link renders as an underlined button that opens the PDF at that evidence.
     - `structure_figure`: the cropped paper figure from `config/paper_figures.json` if present; clicking it jumps to the matching architecture evidence.
     - `layers`: key-module cards from `config/model_layers.json`, each linked to the architecture evidence.
     - For chapters still on the legacy flat-list format (no `innovation` field), the older E01… evidence cards are also rendered below the summary.
   - **Right pane (PDF viewer):**
     - Embeds `/papers/{paper_id}?page=...&evidence=...&embedded=1`, which loads `app/templates/paper_reader.html`.
     - Uses EmbedPDF when `resources/vendor/embedpdf/embedpdf.js` is available; falls back to the browser-native PDF viewer.
     - Runtime annotations are drawn from `config/paper_annotations.json` via `importAnnotations`; the annotation plugin is locked to read-only (`disabledCategories` includes annotation/form/redaction/insert/document-export, and an event guard reverts user edits).
     - Text passages use `HIGHLIGHT` (`type: 9`, multiply blend, opacity 0.5); figure/table boxes use `SQUARE` (`type: 5`, transparent fill, colored stroke).

2. **实验预览 (Notebook Preview)**
   - Always present.
   - Loads `/notebooks/{slug}/content`, which returns an nbconvert HTML export of `notebooks/{slug}.ipynb`.
   - On-demand regeneration: if `notebook_previews/{slug}.html` is missing or older than the `.ipynb`, the endpoint regenerates it with `HTMLExporter(template_name="lab")` and `polish_preview`.
   - `polish_preview` also injects a link-retargeting script: same-origin links to a different path get `target="_top"` (the whole detail page navigates and its hash logic focuses the anchor), external links get `target="_blank"`. The preview iframe sandbox therefore includes `allow-top-navigation-by-user-activation allow-popups`; never let a site-internal link load the application shell inside the iframe.
   - The detail page supports `?embedded=1` (and auto-detects `window.self !== window.top`) to degrade to a content-only view: topbar, sidebar, tabs and pager hidden, only the preview panel fills the viewport — the same convention as the embedded paper reader.
   - Rendered inside a white card with a fixed viewport height.

3. **可交互执行 (Executable Jupyter)**
   - Always present, but disabled for generative chapters when CUDA is unavailable (`chapter_slug == 'generative' and not cuda.available`).
   - When enabled, embeds the JupyterLab URL `{{ jupyter_url }}/lab/tree/notebooks/{slug}.ipynb?token=recsys` on first tab activation (lazy `data-src`).
   - A link to open the same notebook in a new window is provided.

4. **代码实现 (Implementation Source)**
   - Always present.
   - Two-column source browser (`app/source_browser.py`):
     - **Notebook 公用代码**: `recsys_lab/data.py`, `recsys_lab/runtime.py`, `recsys_lab/experiments.py` (for chapter 4 classic), and `tests/test_experiments.py`.
     - **本章节独立目录**: files under `chapter_code/{slug}/` in fixed order (`model.py`, `train.py`, `inference.py`, `test_model.py`, `__init__.py`).
     - **Torch-RecHub 框架源码**: third-party framework modules mapped in `FRAMEWORK_MODULES` (e.g., `torch_rechub.models.matching.dssm`).
   - Syntax highlighting via Pygments with inline line numbers and highlighted landmark lines (`class `, `def forward`, `def run_`, `loss.backward`, `with torch.no_grad`, etc.).
   - A link redirects to the browser-based IDE (`/ide/{slug}`) pointing at `chapter_code/{slug}/train.py:1:1`.

### Responsive rules

- Desktop (`>1050px`): paper-guide split-pane (1:1), source browser (sidebar + editor), preview/execute frames with fixed height.
- Tablet (`<=1050px`): paper guide collapses to a single column with the viewer below the prose; preview frame uses `78vh`.
- Mobile (`<=680px`): tab buttons compact, source browser stacks vertically, frames use viewport height.
- When CUDA is unavailable for generative notebooks, the execute tab button is disabled and shows a tooltip.

### Chapter 3 curriculum and Appendix math map (A.4)

The first eight registered notebooks are the contiguous chapter 3 foundation sequence: the overview, six `kind="curriculum"` lessons (`3_2_data_ml_basics` through `3_7_optimization`), and the `3_8_data_pipeline` data/experiment pipeline. The lessons teach reusable concepts at high-school-graduate level; algorithm notebooks link to their stable anchors and derive only paper-specific mathematics.

`app/content.py:MATH_PREREQUISITES` is the tracked list behind the home page's A.4 数学知识地图. Each entry includes a plain-language intuition, a valid chapter 3 curriculum notebook and anchor, prerequisite topic ids, and the algorithms/models that use it. A.4 is a bounded graph projection, not another course hierarchy: its default view contains the six chapters; focused views contain at most 16 nodes and expand at most two relationship steps. The list must cover every algorithm.

## Tutorial contract

Every algorithm is a separate notebook. A major chapter has an opening/math notebook, algorithm notebooks, and a result summary that reads the recorded experiment JSON. Each algorithm notebook must include paper context, prerequisite math, model structure and formulas, a small Python demonstration, real dataset provenance, training/inference/testing code, metrics, baseline and result discussion. Explain new symbols before using them and keep the prose compact, but never replace a derivation with a framework call.

Classic algorithms may use deterministic MovieLens/Amazon teaching subsets. Deep retrieval, ranking, multitask, Transformer and generative chapters use complete official datasets under `RECSYS_PROFILE=full`; `smoke` is reserved for deterministic CPU/CI validation and must be labeled as such. Never fabricate interactions, exposures, labels or sequences.

Generative notebooks are CUDA-first. Their default runners must reject a host without CUDA; the Web detail page disables interactive execution when CUDA is unavailable. CPU CI may call the explicit `cpu_smoke=True` path only for data contracts, tensor shapes, a bounded forward/backward pass and constrained decoding. Accuracy/effect thresholds for OpenOneRec and HSTU belong in CUDA-gated tests. `Dockerfile` builds the CPU image (CPU-only torch from the locked pytorch-cpu index); `Dockerfile.cuda` builds the CUDA image by swapping in the CUDA-enabled torch wheel from PyPI. `docker-compose.cuda.yml` is an override that switches the affected services to `Dockerfile.cuda` (tagged `recsys-atlas-cuda`) and passes NVIDIA devices into the containers. GitHub Actions builds and publishes both web images plus the IDE image.

## Data and evaluation architecture

### Chapter × dataset matrix

Every algorithm chapter has exactly two dataset tracks, switched by `RECSYS_PROFILE`:

| Chapter | full (paper-aligned) | smoke (CI/teaching adapter) |
|---|---|---|
| 3 (math curriculum) | teaching arrays, no paper protocol | bundled MovieLens latest-small |
| 4.2–4.4, 4.6 classic | MovieLens latest (~33M, `movielens-latest-full`) | latest-small deterministic slice |
| 4.5 GBDT+LR | Criteo_x1 official 7:2:1 (`criteo-x1-full`) | MovieLens-derived CTR slice |
| 5.2 / 5.3 / 5.4 retrieval | Books 5-core / MIND Books 2014 10-core / ML-1M | Amazon 2023 Video Games slice |
| 6.2 DeepFM | Criteo_x1 official 7:2:1 | KuaiRand-Pure |
| 6.3 / 6.4 DIN/DIEN | Electronics 5-core | KuaiRand-Pure |
| 7.2 / 7.3 MMoE/PLE | Census-Income KDD | KuaiRand-Pure |
| 8.2 OpenOneRec | RecIF-Bench authorized copy (`openonerec-recif`, gated) | KuaiRand Semantic-ID adapter |
| 8.3 HSTU | MovieLens 20M (`movielens-20m-full`) | KuaiRand feed sequences |

Summary chapters compare on one shared dataset per family (4: MovieLens, 5: Video Games, 6/7/8: KuaiRand) and never mix both tracks in one table.

### Loader and dispatcher conventions

- Full-profile loaders in `recsys_lab/data.py` read only the manifest resource, verify official statistics and raise on drift. Stable releases (ML-1M, ML-20M, Census-Income, Criteo_x1, RecIF, MIND Books) use exact counts; GroupLens `ml-latest` is regenerated upstream, so its loader asserts a 33M floor and records the actual count in provenance instead of hard-coding one snapshot.
- The official Amazon 5-core files still contain a few users with fewer than 5 interactions. The loader removes only that residual and records `raw_rows` / `dropped_rows` in `frame.attrs`; `test_full_data` asserts the raw line count, the exact residual and that nothing else was dropped.
- MIND's 10-core is **item-first single-pass** (items with ≥10 users, then users with ≥10 items): this reproduces the paper's 6,271,511 samples exactly, while iterating to convergence (4.7M) or filtering users first (5.79M) does not. The paper Table 1 user/item counts (351,356 / 393,801) do not match the reproducible view (218,972 / 369,114), so the sample row count is the comparison anchor and the discrepancy is documented in `mind_amazon_provenance` and the reproduction contract.
- `recsys_lab/runtime.py` `real_*` dispatchers own the full/smoke switch; each chapter `train.py` owns its algorithm-specific full branch, including scale engineering: sparse top-k ItemCF, mini-batched MF/FM training, seeded evaluation samples for quadratic-cost evaluation, and batched full-vocab softmax for HSTU — the `[B, L, V]` logits of a 26k+ vocabulary never materialize in a single forward (the paper's industrial answer is sampled softmax).

### Test and CI tiers

| Tier | Marker / compose service | Trigger | Gate |
|---|---|---|---|
| unittest | `pytest -q` (default suite) | every push, ci.yml `unittest` | API, evidence, content, code quality |
| smoke integration | `docker compose run --rm test` | every push, ci.yml `smoke` | regenerates artifacts, executes all notebooks in smoke mode |
| full data | `pytest -m full_data`, `test-full` | weekly or dispatch | complete official row counts, no truncation |
| CPU accuracy | `pytest -m accuracy`, `test-accuracy` | weekly or dispatch | conservative effect floors on ml-latest / Criteo_x1 |
| CUDA accuracy | `pytest -m cuda_accuracy`, `test-cuda-accuracy` | dispatch, self-hosted GPU | generative gates on RecIF / ML-20M; auto-skipped without CUDA |

Gated resources authenticate through `HF_TOKEN`; image builds receive it as a BuildKit secret (`hf_token`) so it never enters image layers, and both `publish-images.yml` and the dispatch CI jobs inject it from GitHub secrets. `ensure_resources` merges `resources/resource-lock.json` instead of rewriting it, so `--id`-filtered invocations cannot drop the readiness records of other resources.

## Code ownership

Algorithm-specific model, preprocessing, training, inference and tests live in `chapter_code/<slug>/`. Only genuinely cross-chapter data adapters, metrics, seeding and compatibility routes belong in `recsys_lab/`. Generated notebooks are sourced from `scripts/generate_notebooks.py`, `scripts/tutorial_deep_specs.py` and related spec files; update the generator before regenerating artifacts.

### Chapter-code generator (`scripts/generate_chapter_code.py`)

For each notebook slug, the generator writes a small package under `chapter_code/<slug>/`. Checked-in chapter files are the readable source of truth: the generator only scaffolds files that are missing and never overwrites existing ones.

- `model.py`: a minimal executable model for classic chapters (ItemCF, BiasMF, FM, GBDT+LR, SkipGram); for deep/generative chapters it imports the Torch-RecHub class and adds a short structural guide as comments.
- `train.py`: scaffolded only when missing, by extracting the relevant `run_*` functions from `recsys_lab/industrial_experiments.py` and wrapping them in a `train_and_evaluate` entry point; classic chapters delegate to `recsys_lab.experiments.run_classic`. Since `industrial_experiments.py` now keeps only `load_runner` delegation stubs (implementations live in `chapter_code/`), the generator raises a `RuntimeError` instead of emitting an unrunnable stub when the real implementation cannot be extracted — restore the chapter's `train.py` from git in that case.
- `inference.py`: a shared `predict(model, batch)` helper that calls `model.eval()` and `torch.no_grad()`.
- `test_model.py`: a one-epoch smoke test that asserts the training pipeline returns a dict.
- `__init__.py` / `.vscode/settings.json` / `pyrightconfig.json`: package metadata and IDE settings.

The generator also inserts numbered training landmarks ("1) 固定参数初始化…", "2) 按论文结构实例化模型…", etc.) into `train.py` so the source browser highlights the conceptual steps.

### Notebook generator (`scripts/generate_notebooks.py`)

Top-level orchestrator that:

1. Builds the chapter 4 classic notebooks from a single source spec and then renames `3_1_overview` to `4_7_classic_summary`.
2. Merges opening notebooks from `scripts/tutorial_opening_specs.py`.
3. Merges deep/generative notebooks and summaries from `scripts/tutorial_deep_specs.py`.
4. Writes all `notebooks/*.ipynb` files.
5. Sets notebook metadata: `kernelspec`, `language_info`, and `recsys.profile="full"`, `recsys.requires_cuda=true` for chapter 8.

The `notebook(title, goal, source, cells)` helper chooses the dataset based on the title (`dataset_for_title`) and prepends a standard setup cell that loads the real dataset and asserts `randomly_fabricated_rows == 0`. `curriculum_notebook` builds the six chapter 3 concept lessons without pretending their labeled teaching arrays are user behavior. Curriculum setup cells end with `CURRICULUM_FONT_SETUP`: matplotlib's default DejaVu Sans has no CJK glyphs, so charts pick the first available font from Noto CJK (installed via `fonts-noto-cjk` in the images) or common host Chinese fonts — fix the font, never silence missing-glyph warnings with `warnings.filterwarnings`.

### Opening specs (`scripts/tutorial_opening_specs.py`)

Each opening notebook contains:

- A layout/roadmap table showing subchapters, problems, shared math, and typical placement.
- A paper-interpretation section.
- A common-math section with formulas.
- A hands-on NumPy/Matplotlib demo.
- A checks cell.

### Deep specs (`scripts/tutorial_deep_specs.py`)

Each algorithm notebook contains, in order:

1. 学习地图 / problem statement.
2. Paper & Context / source paper summary and reproduction contract.
3. Model Structure & Formula Walkthrough: figure markdown (from `config/paper_figures.json` and `config/model_layers.json`) plus derivation.
4. 公式到代码 / implementation map.
5. Math by Hand: a small NumPy/Matplotlib demo.
6. Data: real dataset provenance and anti-leak checklist.
7. Model & Framework: framework usage notes.
8. Inspect source code cell.
9. Train & Inference: runs the chapter runner and prints the inference contract.
10. Loss curve + metrics bar chart.
11. Paper comparison guard: only computes absolute gaps when `PROFILE == 'full'` and the dataset matches the paper protocol.
12. Test & Results Discussion.
13. Save records to `results/chapter_*/{slug}.json`.
14. Checks and Next Steps.

Summary notebooks add a **Paper Evidence Map** cell that loads `config/paper_evidence.json` and displays the aggregated evidence rows, plus a results aggregation cell, a bar chart, and (for 5/6) a paper-comparability audit table.

## External resources

`config/resources.json` is the tracked source-of-truth. Downloaded papers, full datasets, vendor bundles and `resource-lock.json` live under ignored `resources/`. Run `python scripts/init_resources.py --strict` for required papers/viewer resources. Add `--include-optional --kind datasets --id <id>` for a full dataset. Google Scholar is discovery metadata only: automation must fetch from arXiv, Hugging Face, an official repository or an explicit open/direct URL, and must respect licensing, gating and robots rules.

The initializer is idempotent and standard-library-first. In the application image, `pypdf` additionally validates that PDF title text matches the manifest, preventing an unrelated but syntactically valid PDF from entering the evidence chain. Gated Hugging Face datasets (e.g. `openonerec-recif`) authenticate through the `HF_TOKEN` environment variable; Docker builds receive it as a BuildKit secret (`hf_token`) so it never enters image layers. Docker copies any locally initialized resources and then calls the same initializer; application startup verifies state and downloads only when `RECSYS_RESOURCE_AUTO_DOWNLOAD=1` is explicitly set.

## Paper evidence design

The project uses EmbedPDF rather than PDF-to-HTML. This preserves formula, figure and two-column fidelity and supports programmatic page navigation. `config/paper_evidence.json` maps tutorial keywords to a paper, page, searchable source phrase, quotation and interpretation. Paper guide prose is the first core tab: its text and local PDF reader use equal-width columns, and several underlined semantic annotations per evidence anchor jump the reader to the exact page. Opening/math and summary pages aggregate the relevant chapter evidence instead of showing an empty state. Prefer a cited original paper crop registered in `config/paper_figures.json`; use the generated SVG from `scripts/generate_model_diagrams.py` only as a fallback and never present it as a copied paper figure. The browser-native PDF viewer is the graceful fallback.

### Evidence schema (`config/paper_evidence.json`)

Each chapter key maps to either a legacy list of anchors or an enriched object:

```json
{
  "<chapter_slug>": {
    "items": [
      {
        "id": "unique-anchor-id",
        "paper_id": "<paper-id>",
        "keyword": "中文关键词",
        "label": "原文：标签",
        "page": 1,
        "search": "phrase used to locate the passage",
        "quote": "exact contiguous words from the PDF column",
        "note": "interpretation for the tutorial"
      }
    ],
    "guide": {
      "background": "...",
      "problem": "...",
      "innovation": "...",
      "method": "...",
      "findings": "...",
      "contribution": "...",
      "limitations": "...",
      "one_liner": "..."
    },
    "quick_index": [
      {"label": "...", "page": 1, "evidence_id": "...", "kind": "method"}
    ]
  }
}
```

Guide prose may contain inline links `[[term|page|evidence_id]]`. The `evidence_id` must exist in the same chapter's `items` or in that paper's `paper_annotations.json` ids. Legacy flat-list chapters still render as E01… evidence cards and do not show the summary/guide blocks.

`tests/test_evidence_consistency.py` guards this join: every `quick_index` entry and every prose link must resolve to an item id or an annotation id of the resolved paper (quick_index without an explicit `paper_id` falls back to the chapter's first item's paper, matching `app/evidence.py:paper_guide`), each item's `paper_id` must have an annotations key, and an id present as both item and annotation must agree on `page`. Historical violations are pinned in `KNOWN_ISSUES`; shrink that set when cleaning a chapter, never grow it.

### Runtime annotations (`scripts/build_pdf_annotations.py`)

`config/paper_annotations.json` is generated from `SPECS` in `scripts/build_pdf_annotations.py`. Each spec entry is either:

- `quote` + `col`: a text passage resolved to per-line segment rects by deterministic character-stream matching.
- `rect`: a manually chosen figure/table box in PyMuPDF points (top-left origin).

For two-column papers, constants `L = (50.0, 295.0)` and `R = (305.0, 560.0)` restrict the search to one column. Always run with `--verify` before committing to produce overlay images in `tmp/annotation_overlays/`; the script reports `unresolved=[]` for each paper.

When a SPECS key names the PDF file but the runtime paper id differs (e.g. `matrix-factorization.pdf` vs paper id `mf`), `OUTPUT_ALIASES` maps the SPECS key to the runtime id so the JSON key matches `resources.json` and the `/papers/{id}` routes.

### Paper figures (`config/paper_figures.json`)

Each figure entry declares a paper, page, label, and normalized crop rectangle:

```json
{
  "<paper_id>": {
    "page": 3,
    "label": "Figure 1 · DSSM architecture",
    "crop": [0.05, 0.07, 0.95, 0.35]
  }
}
```

The cropped image is expected at `app/static/paper-figures/{paper_id}.webp` (generated by `scripts/generate_paper_figures.py` / `scripts/embed_paper_figures.py`).

### Model layers (`config/model_layers.json`)

Each paper maps to a list of `[layer_name, explanation]` pairs. These populate the **关键层级** cards in the paper-guide tab and the bullet list under the figure block in generated notebooks.

Paper claims must preserve their original experiment setting. Separate: (1) results reported by the paper, (2) results actually produced by this repository, and (3) expected production impact. Never transfer an online lift, company dataset result or relative improvement to another dataset.

## Verification

Run `docker compose run --rm test`. It regenerates chapter code/notebooks, runs unit/API/effect checks, executes all notebooks in smoke mode and builds previews. Run `docker compose --profile full run --rm test-full` for paper-protocol data/effect validation; it must verify complete test-set counts and must never treat smoke metrics as paper reproduction. Run `docker compose --profile full run --rm test-accuracy` for the CPU accuracy gates on the complete ml-latest / Criteo_x1 datasets, and `docker compose -f docker-compose.yml -f docker-compose.cuda.yml --profile cuda run --rm test-cuda-accuracy` for the generative gates (RecIF-Bench + ML-20M) on a CUDA host; both need `HF_TOKEN` in the environment for gated resources. Ruff and Python compilation are shared with the IDE. When changing paper resources, also run `python scripts/init_resources.py --verify --strict`; when changing the reader, inspect at least one formula/figure page at desktop and narrow viewport.

GitHub Actions runs two workflows: `ci.yml` (unittest + smoke on every push/PR; full-data and CPU accuracy weekly or on dispatch; CUDA accuracy only on self-hosted GPU runners) and `publish-images.yml` (builds and publishes the CPU/CUDA/IDE images, injecting `HF_TOKEN` as a BuildKit secret for gated resources — the token never enters image layers).

The local code-server is passwordless only because Compose binds it to `127.0.0.1`. Keep the Python interpreter at `/usr/local/bin/python`, definition navigation enabled, `train.py` as the default chapter entry, and Secondary Sidebar hidden.
