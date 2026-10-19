from pydantic import BaseModel, Field

# Chroma's rules: 3-63 chars, alphanumeric at both ends. Dots are excluded so
# names can never be ".." or look like an IPv4 address.
COLLECTION_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$"


class ChunkMetadata(BaseModel):
    filename: str
    chunk_index: int
    chunk_text: str
    injection_risk: bool = False


class SkippedFile(BaseModel):
    filename: str
    reason: str


class UploadResponse(BaseModel):
    collection: str
    files_processed: int
    chunks_created: int
    skipped: list[SkippedFile] = []
    phi_entities_pseudonymized: dict[str, int] = {}


class SourceInfo(BaseModel):
    filename: str
    chunk_index: int
    chunk_text: str
    score: float
    injection_risk: bool = False


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    reranking: bool = True
    reidentify: bool = Field(
        False,
        description="Return original PHI values instead of pseudonym tokens. Requires a PHI-access API key.",
    )


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
    phi_protected: bool = False
    reidentified: bool = False
    usage: UsageSummary | None = None


class CollectionInfo(BaseModel):
    name: str
    document_count: int
    phi_protection: bool = False


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., pattern=COLLECTION_NAME_PATTERN)
    phi_protection: bool = Field(
        True,
        description="Pseudonymize PHI before indexing. Secure by default; set false only for non-patient data.",
    )
