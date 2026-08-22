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
# FORWARDED_ALLOW_IPS: uvicorn always wraps ProxyHeadersMiddleware
# (proxy_headers defaults True) but only honours X-Forwarded-For when the
# immediate peer is trusted, and that defaults to 127.0.0.1 alone. HF's ingress
# is not localhost, so without this every client collapsed into one identity:
# the rate limiter keyed on the proxy (measured live -- 10.16.23.249 and two
# siblings), making the 20/min credential cap a shared bucket rather than
# per-client, and audit_logs.ip_hash recorded the ingress node instead of the
# user. Only HF's own infrastructure can open a connection to this container,
# so trusting a private-range peer is safe here.
#
# The CIDR form matters. uvicorn's get_trusted_client_address walks the
# forwarded chain in REVERSE and returns the first untrusted hop, so a header
# a client injects sits to the left of what the proxy appended and is skipped.
# Setting this to "*" would take the opposite path -- always_trust short
# circuits to hosts[0], the client-controlled end -- and hand any caller the
# ability to choose their own rate-limit bucket. Do not "simplify" it to *.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORWARDED_ALLOW_IPS=10.0.0.0/8 \
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
