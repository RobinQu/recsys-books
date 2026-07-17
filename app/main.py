from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.content import CHAPTERS, DATASETS, EVOLUTION, FRAMEWORKS, MODELS, NOTEBOOKS, PRACTICES, SOURCES
from app.source_browser import source_files

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NOTEBOOKS = {
    "3_1_classic_models": "3_1_summary",
    "3_1_overview": "3_1_summary",
    "3_2_retrieval_dssm_mind": "3_2_summary",
    "3_3_ranking_deepfm_din_dien": "3_3_summary",
    "3_4_multitask_mmoe_ple": "3_4_summary",
    "3_0_1_data_and_experiment_pipeline": "3_0_data_pipeline",
    "3_5_0_transformer_foundations": "3_2_0_retrieval_foundations",
    "3_5_1_sasrec": "3_2_3_sasrec",
    "3_5_summary": "3_2_summary",
}
app = FastAPI(title="RecSys Atlas", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")


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
    return {
        "request": request,
        "active": active,
        "evolution": EVOLUTION,
        "models": MODELS,
        "frameworks": FRAMEWORKS,
        "datasets": DATASETS,
        "practices": PRACTICES,
        "notebooks": NOTEBOOKS,
        "chapters": CHAPTERS,
        "sources": SOURCES,
        "course_groups": course_groups,
        "current_notebook_slug": None,
        "jupyter_url": os.getenv("JUPYTER_PUBLIC_URL", "http://localhost:8889"),
        "ide_url": os.getenv("IDE_PUBLIC_URL", "http://localhost:8090"),
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
def notebook_preview(request: Request, slug: str):
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
        path.write_text(body, encoding="utf-8")
        return HTMLResponse(body)
    return FileResponse(path, media_type="text/html")


@app.get("/ide/{slug}")
def open_source_in_ide(slug: str):
    if slug in LEGACY_NOTEBOOKS:
        slug = LEGACY_NOTEBOOKS[slug]
    if slug not in {item["slug"] for item in NOTEBOOKS}:
        raise HTTPException(404, "Unknown notebook")
    ide_url = os.getenv("IDE_PUBLIC_URL", "http://localhost:8090").rstrip("/")
    folder = f"/home/coder/project/chapter_code/{slug}"
    entry = f"{folder}/train.py:1:1"
    return RedirectResponse(f"{ide_url}/?folder={folder}&goto={entry}", status_code=303)


@app.get("/source/{slug}", response_class=HTMLResponse)
def legacy_source_redirect(slug: str):
    """Source is a tab on the chapter detail page, never a separate information page."""
    if slug not in {item["slug"] for item in NOTEBOOKS}:
        raise HTTPException(404, "Unknown notebook")
    return RedirectResponse(f"/notebooks/{slug}#source", status_code=307)


@app.get("/legacy", response_class=HTMLResponse)
def legacy():
    path = ROOT / "legacy_recsys_report.html"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="text/html")


@app.get("/research-ledger")
def research_ledger():
    return FileResponse(ROOT / "recommendation_algorithm_sources.md", media_type="text/markdown; charset=utf-8")


@app.get("/tutorial-requirements")
def tutorial_requirements():
    return FileResponse(ROOT / "TUTORIAL_REQUIREMENTS.md", media_type="text/markdown; charset=utf-8")


@app.exception_handler(404)
def not_found(_: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail or "not found"}, status_code=404)
