from __future__ import annotations

import logging
import os
import ssl

# Corporate proxy SSL bypass — disable SSL verification when behind intercepting proxy
if os.environ.get("DISABLE_SSL_VERIFY") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context
    import requests.adapters
    _original_send = requests.adapters.HTTPAdapter.send
    def _patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return _original_send(self, request, **kwargs)
    requests.adapters.HTTPAdapter.send = _patched_send
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.getLogger(__name__).info("SSL verification disabled (DISABLE_SSL_VERIFY=1)")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from routers import collections, documents, query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(title="ReconRAG API", version="1.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(collections.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
