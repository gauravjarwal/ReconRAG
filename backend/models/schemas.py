from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    filename: str
    chunk_index: int
    chunk_text: str
    injection_risk: bool = False


class UploadResponse(BaseModel):
    collection: str
    files_processed: int
    chunks_created: int


class SourceInfo(BaseModel):
    filename: str
    chunk_index: int
    chunk_text: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    reranking: bool = True


class ActionUsageSummary(BaseModel):
    """Token usage, cost and latency for one pipeline action."""
    action: str
    tokens_in: int
    tokens_out: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    pct_of_total_tokens: float


class UsageSummary(BaseModel):
    """Aggregated token usage across all pipeline actions in a query."""
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    breakdown: list[ActionUsageSummary]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    confidence: float
    model_used: str
    skipped_generation: bool
    injection_detected: bool = False
    usage: UsageSummary | None = None


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")
