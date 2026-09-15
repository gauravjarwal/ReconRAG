import logging

from fastapi import APIRouter, HTTPException

from models.schemas import ActionUsageSummary, QueryRequest, QueryResponse, UsageSummary
from pipelines.output_validation import validate_output
from pipelines.security import sanitize_query
from pipelines.usage import aggregate_usage, ActionUsage
from services.generation import generate
from services.retrieval import retrieve

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


def _build_usage_summary(action_usages: list[ActionUsage]) -> UsageSummary:
    data = aggregate_usage(action_usages)
    return UsageSummary(
        total_tokens=data["total_tokens"],
        total_cost_usd=data["total_cost_usd"],
        total_latency_ms=data["total_latency_ms"],
        breakdown=[ActionUsageSummary(**item) for item in data["breakdown"]],
    )


@router.post("/api/collections/{collection_name}/query", response_model=QueryResponse)
async def query_collection(collection_name: str, body: QueryRequest) -> QueryResponse:
    # Security: sanitize and detect injection
    clean_question, injection_detected = sanitize_query(body.question)

    # Retrieval (embed + vector search + BM25 + rerank)
    sources, confidence, retrieval_usages = retrieve(collection_name, clean_question, reranking=body.reranking)

    if not sources:
        usage = _build_usage_summary(retrieval_usages)
        return QueryResponse(
            answer="I could not find sufficient relevant information in the collection to answer this question.",
            sources=[],
            confidence=confidence,
            model_used="none",
            skipped_generation=True,
            injection_detected=injection_detected,
            usage=usage,
        )

    # Generation (Claude primary → Gemini fallback)
    try:
        raw_answer, model_used, gen_usage = generate(clean_question, sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Output validation
    sources_dicts = [s.model_dump() for s in sources]
    validated_answer = validate_output(raw_answer, sources_dicts)

    # Aggregate usage across all actions
    all_usages = retrieval_usages + [gen_usage]
    usage = _build_usage_summary(all_usages)

    return QueryResponse(
        answer=validated_answer,
        sources=sources,
        confidence=confidence,
        model_used=model_used,
        skipped_generation=False,
        injection_detected=injection_detected,
        usage=usage,
    )
