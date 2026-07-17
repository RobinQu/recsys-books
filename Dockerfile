FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt

COPY . .
RUN python scripts/generate_notebooks.py && python scripts/build_previews.py

EXPOSE 8000 8888
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

