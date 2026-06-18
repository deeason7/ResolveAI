# Single-container image for a Hugging Face Space (Docker SDK).
#
# A free Space gives you exactly one container, so the API and both Redis-stream
# workers run side by side here under supervisord (see supervisord.conf): each
# program restarts independently, so a crashed worker never takes the API down
# with it. The multi-container topology for local / VM use lives in
# docker-compose.yml and docker-compose.prod.yml — this file is only for the
# free deployment, where Postgres/Redis/Qdrant/Neo4j are all managed elsewhere.
FROM python:3.11-slim

# HF routes external traffic to app_port (declared in the README front-matter);
# 7860 is its default. uvicorn binds the same port via supervisord.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Torch arrives transitively (via sentence-transformers), and pip's default
# wheel is the CUDA build — it bundles the multi-GB nvidia-* / triton /
# cuda-toolkit stack, every byte of it useless on a CPU-only Space. Install the
# CPU wheel first, from PyTorch's own CPU index, so the editable install below
# finds torch already satisfied and never reaches for the CUDA one. This stays
# in the Dockerfile (not pyproject) so local/GPU dev still resolves torch
# normally — it's a deploy-only concern. It's also the heaviest, most stable
# layer, so it sits above the pyproject COPY to cache independently of dep edits.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Then the rest of the deps, from pyproject alone, so this layer caches across
# code changes. The "space" extra adds supervisor on top of the runtime deps.
COPY backend/pyproject.toml ./
RUN pip install -e ".[space]"

# Application code, then the process-supervisor config.
COPY backend/ ./
COPY supervisord.conf /etc/supervisor/supervisord.conf

# HF Spaces run the container as a non-root uid 1000. Own the app dir and the
# model cache so sentence-transformers downloads and supervisor logs can write.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -sf "http://localhost:${PORT}/api/v1/health" || exit 1

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
