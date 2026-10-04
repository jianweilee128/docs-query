FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Install deps first so code edits do not bust the layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY rag/ rag/
COPY config/ config/
COPY prompts/ prompts/
COPY data/ data/
COPY eval/ eval/
COPY main.py docker/entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Chroma lives on a volume; first start runs ingest if the index is empty.
VOLUME ["/app/chroma"]

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "main.py"]
