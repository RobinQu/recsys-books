FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl fonts-noto-cjk poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Locked dependency layer: only rebuilds when pyproject.toml or uv.lock changes.
# The environment lives outside /workspace so the compose bind mount cannot shadow it.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .
ARG RECSYS_RESOURCE_INIT=download
RUN if [ "$RECSYS_RESOURCE_INIT" = "download" ]; then python scripts/init_resources.py --strict; else python scripts/init_resources.py --verify; fi \
    && python scripts/generate_model_diagrams.py \
    && python scripts/generate_paper_figures.py \
    && python scripts/generate_chapter_code.py \
    && python scripts/generate_notebooks.py \
    && python scripts/build_previews.py

EXPOSE 8000 8888
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
