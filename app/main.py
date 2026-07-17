from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.content import CHAPTERS, DATASETS, EVOLUTION, FRAMEWORKS, MODELS, NOTEBOOKS, PRACTICES, SOURCES

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NOTEBOOKS = {
    "3_1_classic_models": "3_1_summary",
    "3_1_overview": "3_1_summary",
    "3_2_retrieval_dssm_mind": "3_2_summary",
    "3_3_ranking_deepfm_din_dien": "3_3_summary",
    "3_4_multitask_mmoe_ple": "3_4_summary",
}
app = FastAPI(title="RecSys Atlas", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")


def page_context(request: Request, active: str = "overview") -> dict:
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
        "jupyter_url": os.getenv("JUPYTER_PUBLIC_URL", "http://localhost:8889"),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", page_context(request))


@app.get("/chapters/{slug}", response_class=HTMLResponse)
def chapter_detail(request: Request, slug: str):
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
        "chapter_slug": chapter_slug,
        "chapter": CHAPTERS.get(chapter_slug) if chapter_slug else None,
        "chapter_opening": next((n for n in NOTEBOOKS if chapter_slug and n["slug"] == CHAPTERS[chapter_slug]["opening"]), None),
        "chapter_sequence": chapter_sequence,
        "previous_notebook": NOTEBOOKS[notebook_index - 1] if notebook_index > 0 else None,
        "next_notebook": NOTEBOOKS[notebook_index + 1] if notebook_index + 1 < len(NOTEBOOKS) else None,
    })
    return templates.TemplateResponse(request, "notebook_shell.html", context)


@app.get("/notebooks/{slug}/content", response_class=HTMLResponse)
def notebook_preview_content(slug: str):
    if slug in LEGACY_NOTEBOOKS:
        return RedirectResponse(f"/notebooks/{LEGACY_NOTEBOOKS[slug]}/content", status_code=307)
    if slug not in {n["slug"] for n in NOTEBOOKS}:
        raise HTTPException(404, "Unknown notebook")
    path = ROOT / "notebook_previews" / f"{slug}.html"
    if not path.exists():
        return HTMLResponse("<h1>预览尚未生成</h1><p>运行 <code>python scripts/build_previews.py</code></p>", 503)
    return FileResponse(path, media_type="text/html")


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
