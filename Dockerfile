# API and worker share one image. Models are baked in, so containers start with no network.
FROM python:3.12-slim AS build
RUN pip install --no-cache-dir "uv>=0.5"
WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-default-groups
COPY src ./src
COPY README.md* ./
RUN uv sync --frozen --no-dev --no-default-groups
RUN uv pip install --python .venv "spacy>=3.8" "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
ENV MODEL_CACHE_DIR=/app/models
RUN .venv/bin/python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
from reconrag.config import Settings; \
from reconrag.ingestion.ocr import _engine; \
s = Settings(_env_file=None); \
TextEmbedding(s.embedding_model, cache_dir=str(s.model_cache_dir)); \
TextCrossEncoder(s.cross_encoder_model, cache_dir=str(s.model_cache_dir)); \
_engine()"
# huggingface_hub writes its cache index as 0600, and the image runs as a non-root user.
RUN chmod -R a+rX /app/models

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY --from=build /app/src ./src
COPY --from=build /app/models ./models
COPY alembic.ini ./
COPY migrations ./migrations
RUN mkdir -p /data/blobs && chown -R app:app /data
ENV PATH=/app/.venv/bin:$PATH MODEL_CACHE_DIR=/app/models BLOB_DIR=/data/blobs PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
USER app
EXPOSE 8000
CMD ["reconrag", "api", "--host", "0.0.0.0", "--port", "8000"]
