from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.content import (
    CHAPTERS,
    DATASETS,
    EVOLUTION,
    MATH_PREREQUISITES,
    MODELS,
    NOTEBOOK_KIND_LABELS,
    NOTEBOOKS,
    PRACTICES,
    SOURCES,
    notebook_has_paper_guide,
)
from app.evidence import paper_guide, paper_payload, source_paper_links
from app.knowledge_graph import build_knowledge_graph
from app.notebook_preview import polish_preview
from app.source_browser import chapter_source_slug, source_files
from recsys_lab.resources import RESOURCE_ROOT, ensure_resources

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NOTEBOOKS = {
    "3_1_classic_models": "3_1_summary",
    "3_1_overview": "3_1_summary",
    "3_2_retrieval_dssm_mind": "3_2_summary",
    "3_3_ranking_deepfm_din_dien": "3_3_summary",
    "3_4_multitask_mmoe_ple": "3_4_summary",
    "3_0_1_data_and_experiment_pipeline": "3_0_7_data_pipeline",
    "3_0_data_pipeline": "3_0_7_data_pipeline",
    "4_1_generative_overview": "4_3_generative_summary",
    "4_2_openonerec_practice": "4_1_openonerec_practice",
    "4_3_dlrm_hstu_practice": "4_2_dlrm_hstu_practice",
    "4_4_generative_summary": "4_3_generative_summary",
    "a_4_1_data_ml_basics": "3_0_1_data_ml_basics",
    "a_4_2_linear_algebra": "3_0_2_linear_algebra",
    "a_4_3_calculus": "3_0_3_calculus",
    "a_4_4_probability_statistics": "3_0_4_probability_statistics",
    "a_4_5_optimization": "3_0_6_optimization",
    "a_4_6_information_theory": "3_0_5_information_theory",
}
@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.resources = ensure_resources(
        download=os.getenv("RECSYS_RESOURCE_AUTO_DOWNLOAD", "0") == "1",
        strict=os.getenv("RECSYS_RESOURCE_STRICT", "0") == "1",
        offline=os.getenv("RECSYS_RESOURCE_OFFLINE", "0") == "1",
    )
    yield


RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="RecSys Atlas", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
app.mount("/resources", StaticFiles(directory=RESOURCE_ROOT, check_dir=False), name="resources")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")


def cuda_status() -> dict:
    """Report the CUDA capability visible to the Web/Jupyter container."""
    forced = os.getenv("RECSYS_CUDA_AVAILABLE")
    if forced is not None:
        available = forced.casefold() in {"1", "true", "yes", "on"}
        return {"available": available, "devices": 1 if available else 0, "name": "configured CUDA" if available else None}
    try:
        import torch

        available = torch.cuda.is_available()
        return {
            "available": available,
            "devices": torch.cuda.device_count() if available else 0,
            "name": torch.cuda.get_device_name(0) if available else None,
        }
    except (ImportError, RuntimeError):
        return {"available": False, "devices": 0, "name": None}


def page_context(request: Request, active: str = "overview") -> dict:
    course_groups = []
    for key, chapter in CHAPTERS.items():
        ordered_slugs = list(dict.fromkeys([chapter["opening"], *chapter["notebooks"], chapter["summary"]]))
        course_groups.append({
            "key": key,
            "number": chapter["number"],
            "title": chapter["title"],
            "items": [item for slug in ordered_slugs for item in NOTEBOOKS if item["slug"] == slug],
        })
    knowledge_graph = {
        "default": build_knowledge_graph(CHAPTERS, MODELS, MATH_PREREQUISITES),
        "views": {
            f"chapter:{key}": build_knowledge_graph(
                CHAPTERS, MODELS, MATH_PREREQUISITES, focus_id=f"chapter:{key}"
            )
            for key in CHAPTERS
        },
    }
    return {
        "request": request,
        "active": active,
        "evolution": EVOLUTION,
        "models": MODELS,
        "datasets": DATASETS,
        "practices": PRACTICES,
        "notebooks": NOTEBOOKS,
        "notebook_kind_labels": NOTEBOOK_KIND_LABELS,
        "math_prerequisites": MATH_PREREQUISITES,
        "knowledge_graph": knowledge_graph,
        "chapters": CHAPTERS,
        "sources": SOURCES,
        "source_paper_links": source_paper_links(),
        "course_groups": course_groups,
        "current_notebook_slug": None,
        "jupyter_url": os.getenv("JUPYTER_PUBLIC_URL", "http://localhost:8889"),
        "ide_url": os.getenv("IDE_PUBLIC_URL", "http://localhost:8090"),
        "cuda": cuda_status(),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", page_context(request))


@app.get("/chapters/{slug}")
def legacy_chapter_redirect(slug: str):
    """Compatibility only: categories live on the home page, not on a third page level."""
    if slug == "transformer":
        return RedirectResponse("/notebooks/3_2_0_retrieval_foundations", status_code=307)
    chapter = CHAPTERS.get(slug)
    if chapter is None:
        raise HTTPException(404, "Unknown chapter")
    return RedirectResponse(f"/notebooks/{chapter['opening']}", status_code=307)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "notebooks": len(NOTEBOOKS), "models": len(MODELS)}


@app.get("/api/resources")
def resource_status(request: Request):
    return getattr(request.app.state, "resources", ensure_resources(download=False))


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
def paper_reader(
    request: Request,
    paper_id: str,
    page: int = Query(default=1, ge=1),
    evidence: str | None = None,
    embedded: bool = False,
):
    paper = paper_payload(paper_id)
    if paper is None:
        raise HTTPException(404, "Unknown paper")
    selected = next((item for item in paper["anchors"] if item["id"] == evidence), None)
    if selected is None and evidence:
        # A runtime annotation id (quick-index card or prose keyword) can also
        # drive navigation: jump to its page so the reader flashes that region.
        annotation = next((a for a in paper["annotations"] if a["id"] == evidence), None)
        if annotation:
            selected = {"id": annotation["id"], "page": annotation["page"],
                        "label": annotation["id"], "search": "", "quote": "", "note": ""}
    if selected:
        page = selected["page"]
    context = page_context(request, "sources")
    context.update({"paper": paper, "paper_page": page, "selected_evidence": selected, "embedded": embedded})
    return templates.TemplateResponse(request, "paper_reader.html", context)


@app.get("/api/papers/{paper_id}")
def paper_api(paper_id: str):
    paper = paper_payload(paper_id)
    if paper is None:
        raise HTTPException(404, "Unknown paper")
    return paper


@app.get("/api/models")
def models(stage: str | None = None):
    rows = MODELS if not stage else [m for m in MODELS if stage.lower() in m["stage"].lower()]
    return {"items": rows, "count": len(rows)}


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=80)):
    needle = q.casefold()
    rows = [m for m in MODELS if needle in " ".join(str(v) for v in m.values()).casefold()]
    source_rows = [{"name": n, "url": u, "type": t} for n, u, t in SOURCES if needle in f"{n} {t}".casefold()]
    return {"models": rows, "sources": source_rows}


@app.get("/notebooks/{slug}", response_class=HTMLResponse)
def notebook_preview(request: Request, slug: str, embedded: bool = False):
    if slug in LEGACY_NOTEBOOKS:
        return RedirectResponse(f"/notebooks/{LEGACY_NOTEBOOKS[slug]}", status_code=307)
    notebook_index = next((index for index, item in enumerate(NOTEBOOKS) if item["slug"] == slug), None)
    if notebook_index is None:
        raise HTTPException(404, "Unknown notebook")
    notebook = NOTEBOOKS[notebook_index]
    chapter_slug = next(
        (key for key, chapter in CHAPTERS.items() if slug in chapter["notebooks"] or slug in {chapter["summary"], chapter["opening"]}),
        None,
    )
    context = page_context(request, chapter_slug or "labs")
    chapter_sequence = []
    if chapter_slug:
        chapter_data = CHAPTERS[chapter_slug]
        ordered_slugs = list(dict.fromkeys([chapter_data["opening"], *chapter_data["notebooks"], chapter_data["summary"]]))
        chapter_sequence = [n for slug_value in ordered_slugs for n in NOTEBOOKS if n["slug"] == slug_value]
    context.update({
        "notebook": notebook,
        "current_notebook_slug": slug,
        "chapter_slug": chapter_slug,
        "chapter": CHAPTERS.get(chapter_slug) if chapter_slug else None,
        "chapter_sequence": chapter_sequence,
        "previous_notebook": NOTEBOOKS[notebook_index - 1] if notebook_index > 0 else None,
        "next_notebook": NOTEBOOKS[notebook_index + 1] if notebook_index + 1 < len(NOTEBOOKS) else None,
        "source_groups": source_files(slug),
        "paper_guide": paper_guide(slug),
        "show_paper_guide": notebook_has_paper_guide(slug),
        "embedded": embedded,
    })
    return templates.TemplateResponse(request, "notebook_shell.html", context)


@app.get("/notebooks/{slug}/content", response_class=HTMLResponse)
def notebook_preview_content(slug: str):
    if slug in LEGACY_NOTEBOOKS:
        return RedirectResponse(f"/notebooks/{LEGACY_NOTEBOOKS[slug]}/content", status_code=307)
    if slug not in {n["slug"] for n in NOTEBOOKS}:
        raise HTTPException(404, "Unknown notebook")
    path = ROOT / "notebook_previews" / f"{slug}.html"
    notebook_path = ROOT / "notebooks" / f"{slug}.ipynb"
    if not notebook_path.exists():
        raise HTTPException(404, "Notebook file is missing")
    # A generated notebook can be newer than a checked-in preview. Regenerate
    # that one document on demand so readers never see stale/missing content.
    if not path.exists() or path.stat().st_mtime < notebook_path.stat().st_mtime:
        config = Config(); config.HTMLExporter.exclude_input_prompt = True; config.HTMLExporter.exclude_output_prompt = True
        exporter = HTMLExporter(template_name="lab", config=config)
        body, _ = exporter.from_notebook_node(nbformat.read(notebook_path, as_version=4))
        path.parent.mkdir(exist_ok=True)
        body = polish_preview(body)
        path.write_text(body, encoding="utf-8")
        return HTMLResponse(body)
    return FileResponse(path, media_type="text/html")


@app.get("/ide/{slug}")
def open_source_in_ide(slug: str):
    if slug in LEGACY_NOTEBOOKS:
        slug = LEGACY_NOTEBOOKS[slug]
    notebook = next((item for item in NOTEBOOKS if item["slug"] == slug), None)
    if notebook is None:
        raise HTTPException(404, "Unknown notebook")
    ide_url = os.getenv("IDE_PUBLIC_URL", "http://localhost:8090").rstrip("/")
    if notebook["kind"] == "curriculum":
        folder = "/home/coder/project"
        entry = f"{folder}/scripts/tutorial_math_specs.py:1:1"
        return RedirectResponse(f"{ide_url}/?folder={folder}&goto={entry}", status_code=303)
    folder = f"/home/coder/project/chapter_code/{chapter_source_slug(slug)}"
    entry = f"{folder}/train.py:1:1"
    return RedirectResponse(f"{ide_url}/?folder={folder}&goto={entry}", status_code=303)


@app.get("/source/{slug}", response_class=HTMLResponse)
def legacy_source_redirect(slug: str):
    """Source is a tab on the chapter detail page, never a separate information page."""
    if slug in LEGACY_NOTEBOOKS:
        slug = LEGACY_NOTEBOOKS[slug]
    notebook = next((item for item in NOTEBOOKS if item["slug"] == slug), None)
    if notebook is None:
        raise HTTPException(404, "Unknown notebook")
    return RedirectResponse(f"/notebooks/{slug}#source", status_code=307)


@app.get("/research-ledger")
def research_ledger():
    return FileResponse(ROOT / "docs" / "recommendation_algorithm_sources.md", media_type="text/markdown; charset=utf-8")


@app.get("/tutorial-requirements")
def tutorial_requirements():
    return FileResponse(ROOT / "docs" / "TUTORIAL_REQUIREMENTS.md", media_type="text/markdown; charset=utf-8")


@app.exception_handler(404)
def not_found(_: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail or "not found"}, status_code=404)
