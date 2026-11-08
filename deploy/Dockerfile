# ── Stage 1: Build ────────────────────────────────────
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/

# ── Stage 2: Runtime ──────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# C11: runtime state lives in a mounted volume, NOT in image layers —
# the old `COPY data/` baked org memory DBs into the image. The app
# creates data dirs on demand (MemoryStore mkdirs parents).
RUN mkdir -p /app/data/memory && chown -R appuser:appuser /app/data
VOLUME /app/data/memory

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.web:app", "--host", "0.0.0.0", "--port", "8000"]
