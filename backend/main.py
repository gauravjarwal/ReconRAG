from __future__ import annotations

import logging
import os
import ssl
from contextlib import asynccontextmanager

from observability import RequestContextMiddleware, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Corporate proxy SSL bypass — disable SSL verification when behind intercepting proxy.
# Never enable this in production: it allows anyone on the network path to read PHI in transit.
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
    logger.warning("SSL verification disabled (DISABLE_SSL_VERIFY=1) — do not use with real PHI")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import settings
from pipelines import phi
from rate_limit import limiter
from routers import collections, documents, query
from services.deidentification import init_vault


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail closed: if PHI tooling can't initialise, don't serve requests.
    init_vault()
    if settings.phi_ner_enabled:
        phi.load_ner(settings.phi_ner_model)
    else:
        logger.warning("PHI_NER_ENABLED=false: names and locations will NOT be de-identified")
    yield


app = FastAPI(title="ReconRAG API", version="1.1.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)
# Added last so it runs first: every response, including errors, gets an ID and headers.
app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_size_bytes + 1024 * 1024)

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(collections.router)


@app.get("/api/health")
@limiter.exempt
def health() -> dict:
    return {"status": "ok"}
