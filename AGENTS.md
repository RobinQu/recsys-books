# RecSys Atlas engineering guide

## Product structure

Keep exactly two information levels: the home page and an algorithm/notebook detail page. Do not add category landing pages. The global top bar and left navigation are shared templates; a detail page only changes active state. The home page ends with one Appendix chapter containing paper, dataset, and Notebook subchapters; do not restore a framework-selection section. Every detail page has exactly four core tabs: paper guide, experiment preview, executable Jupyter, and implementation source.

## Tutorial contract

Every algorithm is a separate notebook. A major chapter has an opening/math notebook, algorithm notebooks, and a result summary that reads the recorded experiment JSON. Each algorithm notebook must include paper context, prerequisite math, model structure and formulas, a small Python demonstration, real dataset provenance, training/inference/testing code, metrics, baseline and result discussion. Explain new symbols before using them and keep the prose compact, but never replace a derivation with a framework call.

Classic algorithms may use deterministic MovieLens/Amazon teaching subsets. Deep retrieval, ranking, multitask, Transformer and generative chapters use complete official datasets under `RECSYS_PROFILE=full`; `smoke` is reserved for deterministic CPU/CI validation and must be labeled as such. Never fabricate interactions, exposures, labels or sequences.

Generative notebooks are CUDA-first. Their default runners must reject a host without CUDA; the Web detail page disables interactive execution when CUDA is unavailable. CPU CI may call the explicit `cpu_smoke=True` path only for data contracts, tensor shapes, a bounded forward/backward pass and constrained decoding. Accuracy/effect thresholds for OpenOneRec and HSTU belong in CUDA-gated tests. Use `docker-compose.cuda.yml` to pass NVIDIA devices into containers.

## Code ownership

Algorithm-specific model, preprocessing, training, inference and tests live in `chapter_code/<slug>/`. Only genuinely cross-chapter data adapters, metrics, seeding and compatibility routes belong in `recsys_lab/`. Generated notebooks are sourced from `scripts/generate_notebooks.py`, `scripts/tutorial_deep_specs.py` and related spec files; update the generator before regenerating artifacts.

## External resources

`config/resources.json` is the tracked source-of-truth. Downloaded papers, full datasets, vendor bundles and `resource-lock.json` live under ignored `resources/`. Run `python scripts/init_resources.py --strict` for required papers/viewer resources. Add `--include-optional --kind datasets --id <id>` for a full dataset. Google Scholar is discovery metadata only: automation must fetch from arXiv, Hugging Face, an official repository or an explicit open/direct URL, and must respect licensing, gating and robots rules.

The initializer is idempotent and standard-library-first. In the application image, `pypdf` additionally validates that PDF title text matches the manifest, preventing an unrelated but syntactically valid PDF from entering the evidence chain. Docker copies any locally initialized resources and then calls the same initializer; application startup verifies state and downloads only when `RECSYS_RESOURCE_AUTO_DOWNLOAD=1` is explicitly set.

## Paper evidence design

The project uses EmbedPDF rather than PDF-to-HTML. This preserves formula, figure and two-column fidelity and supports programmatic page navigation. `config/paper_evidence.json` maps tutorial keywords to a paper, page, searchable source phrase, quotation and interpretation. Paper guide prose is the first core tab: its text and local PDF reader use equal-width columns, and several underlined semantic annotations per evidence anchor jump the reader to the exact page. Opening/math and summary pages aggregate the relevant chapter evidence instead of showing an empty state. Prefer a cited original paper crop registered in `config/paper_figures.json`; use the generated SVG from `scripts/generate_model_diagrams.py` only as a fallback and never present it as a copied paper figure. The browser-native PDF viewer is the graceful fallback.

Paper claims must preserve their original experiment setting. Separate: (1) results reported by the paper, (2) results actually produced by this repository, and (3) expected production impact. Never transfer an online lift, company dataset result or relative improvement to another dataset.

## Verification

Run `docker compose run --rm test`. It regenerates chapter code/notebooks, runs unit/API/effect checks, executes all notebooks in smoke mode and builds previews. Run `docker compose --profile full run --rm test-full` for paper-protocol data/effect validation; it must verify complete test-set counts and must never treat smoke metrics as paper reproduction. Ruff and Python compilation are shared with the IDE. When changing paper resources, also run `python scripts/init_resources.py --verify --strict`; when changing the reader, inspect at least one formula/figure page at desktop and narrow viewport.

The local code-server is passwordless only because Compose binds it to `127.0.0.1`. Keep the Python interpreter at `/usr/local/bin/python`, definition navigation enabled, `train.py` as the default chapter entry, and Secondary Sidebar hidden.
