FROM python:3.11-slim

# libmagic for python-magic MIME detection
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user. Data dirs are created here so named volumes
# inherit this ownership on first mount; the vault dir is owner-only.
RUN useradd --system --uid 10001 --home /app app \
    && mkdir -p /app/chroma_data /app/phi_vault \
    && chmod 700 /app/phi_vault \
    && chown -R app:app /app/chroma_data /app/phi_vault
USER app

ENV PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False

EXPOSE 8000

# Single worker: the BM25 cache and its invalidation are in-process.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
