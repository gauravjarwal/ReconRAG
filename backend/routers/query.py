import logging

from chromadb import Collection
from fastapi import APIRouter, Depends, HTTPException, Request

from auth import Principal, require_api_key
from config import settings
from models.schemas import ActionUsageSummary, QueryRequest, QueryResponse, SourceInfo, UsageSummary
from observability import audit
from pipelines.output_validation import validate_output
from pipelines.retry import UpstreamServiceError
from pipelines.security import sanitize_query
from pipelines.usage import aggregate_usage, ActionUsage
from pipelines.vector_store import is_phi_protected
from rate_limit import limiter
from routers.dependencies import existing_collection
from services.deidentification import pseudonymize_query, reidentify
from services.generation import generate
from services.retrieval import retrieve

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

NO_ANSWER = "I could not find sufficient relevant information in the collection to answer this question."


def _build_usage_summary(action_usages: list[ActionUsage]) -> UsageSummary:
    data = aggregate_usage(action_usages)
    return UsageSummary(
        total_tokens=data["total_tokens"],
        total_cost_usd=data["total_cost_usd"],
        total_latency_ms=data["total_latency_ms"],
        breakdown=[ActionUsageSummary(**item) for item in data["breakdown"]],
    )


# Sync def: FastAPI runs it in a threadpool, so blocking model calls don't stall the event loop.
@router.post("/api/collections/{collection_name}/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit_query)
def query_collection(
    request: Request,
    body: QueryRequest,
    principal: Principal = Depends(require_api_key),
    collection: Collection = Depends(existing_collection),
) -> QueryResponse:
    phi_protected = is_phi_protected(collection)
    reidentified = phi_protected and body.reidentify

    if body.reidentify and not principal.can_reidentify:
        audit("reidentify_denied", key_id=principal.key_id, collection=collection.name)
        raise HTTPException(status_code=403, detail="This API key is not authorised to re-identify PHI.")

    # PHI: pseudonymize the question with the collection's keys so it matches
    # the pseudonymized index, and so no raw PHI reaches the model APIs.
    question = body.question
    phi_in_query = 0
    if phi_protected:
        pseudo = pseudonymize_query(collection.name, question)
        question, phi_in_query = pseudo.text, sum(pseudo.entity_counts.values())

    # Security: detect injection. Retrieval uses the plain (truncated) question;
    # the neutralizing frame is for the generator only and would pollute search.
    retrieval_question = question[: settings.max_query_length]
    clean_question, injection_detected = sanitize_query(question)

    try:
        sources, confidence, retrieval_usages = retrieve(collection, retrieval_question, reranking=body.reranking)
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    answer, model_used, all_usages = NO_ANSWER, "none", retrieval_usages
    if sources:
        try:
            raw_answer, model_used, gen_usage = generate(clean_question, sources)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        answer = validate_output(raw_answer, [s.model_dump() for s in sources])
        all_usages = retrieval_usages + [gen_usage]

    if reidentified:
        answer = reidentify(collection.name, answer)
        sources = [
            SourceInfo(**{**s.model_dump(), "chunk_text": reidentify(collection.name, s.chunk_text)})
            for s in sources
        ]

    audit(
        "query",
        key_id=principal.key_id,
        collection=collection.name,
        phi_protection=phi_protected,
        phi_entities_in_query=phi_in_query,
        reidentified=reidentified,
        injection_detected=injection_detected,
        model_used=model_used,
        sources=len(sources),
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        model_used=model_used,
        skipped_generation=not sources,
        injection_detected=injection_detected,
        phi_protected=phi_protected,
        reidentified=reidentified,
        usage=_build_usage_summary(all_usages),
    )
