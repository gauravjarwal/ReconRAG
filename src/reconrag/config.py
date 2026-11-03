"""Typed settings. Every tunable lives here, so a value can't be guessed twice."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str) -> frozenset[str]:
    return frozenset(item.strip() for item in value.split(",") if item.strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Infrastructure
    database_url: str = "postgresql://reconrag:reconrag@localhost:5433/reconrag"
    db_pool_min: int = 2
    db_pool_max: int = 20
    db_pool_timeout_s: float = 5.0
    db_statement_timeout_ms: int = 15_000
    redis_url: str = "redis://localhost:6380/0"
    redis_timeout_s: float = 0.25
    blob_dir: Path = Path("data/blobs")

    # Models
    embedding_provider: Literal["local", "gemini"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    model_cache_dir: Path = Path("data/models")
    reranker: Literal["none", "cross-encoder", "llm"] = "cross-encoder"
    cross_encoder_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    google_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    claude_model: str = "claude-haiku-5-5"
    # First available provider answers; the next takes over only if the first is unavailable.
    provider_order: str = "gemini,anthropic"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 2048
    breaker_failure_threshold: int = 5
    breaker_cooldown_s: float = 30.0

    # Ingestion
    max_file_bytes: int = 50 * 1024 * 1024
    max_files_per_request: int = 100
    max_pages: int = 2_000
    max_unzipped_bytes: int = 200 * 1024 * 1024
    max_zip_ratio: int = 100
    max_spreadsheet_rows: int = 100_000
    ocr_enabled: bool = True
    ocr_dpi: int = 200
    min_page_chars: int = 40
    quality_flag_below: float = 0.6
    quality_reject_below: float = 0.15
    near_duplicate_bits: int = 3
    chunk_target_tokens: int = 300
    chunk_overlap_tokens: int = 40
    chunk_min_tokens: int = 25
    table_max_tokens: int = 450

    # Worker
    worker_poll_s: float = 1.0
    worker_lease_s: float = 60.0
    worker_heartbeat_s: float = 15.0
    worker_max_attempts: int = 3
    worker_parse_timeout_s: float = 300.0
    worker_metrics_port: int = 0

    # Retrieval
    dense_top_k: int = 30
    lexical_top_k: int = 30
    fuzzy_top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    fuzzy_weight: float = 0.5
    rerank_candidates: int = 20
    rerank_concurrency: int = 2
    rerank_threads: int = 2
    rerank_wait_s: float = 5.0
    final_top_k: int = 5
    hnsw_ef_search: int = 100
    hnsw_max_scan_tuples: int = 20_000
    query_cache_ttl_s: int = 3_600
    context_max_tokens: int = 2_500
    expand_neighbours: int = 1
    # Uncalibrated until the evals are adapted to v2. Below these the service refuses to answer.
    gate_min_dense: float = 0.30
    gate_min_rerank: float = -6.0

    # API
    api_keys: str = ""
    phi_access_api_keys: str = ""
    cors_origins: str = "http://localhost:8501"
    max_request_bytes: int = 512 * 1024 * 1024
    max_question_chars: int = 2_000
    max_answer_chars: int = 6_000
    rate_limit_query_per_min: int = 30
    rate_limit_upload_per_min: int = 30
    rate_limit_default_per_min: int = 300

    # PHI
    phi_hmac_key: str = ""
    phi_encryption_key: str = ""
    phi_ner_enabled: bool = False
    phi_ner_model: str = "en_core_web_lg"

    log_level: str = "INFO"
    log_json: bool = True
    environment: Literal["dev", "test", "prod"] = "dev"
    disable_auth: bool = Field(default=False, description="Local development only.")

    @property
    def api_key_set(self) -> frozenset[str]:
        return _csv(self.api_keys)

    @property
    def phi_key_set(self) -> frozenset[str]:
        return _csv(self.phi_access_api_keys)

    @property
    def cors_origin_list(self) -> list[str]:
        return sorted(_csv(self.cors_origins))

    @property
    def providers(self) -> list[str]:
        return [name.strip() for name in self.provider_order.split(",") if name.strip()]

    @property
    def phi_available(self) -> bool:
        """PHI collections need both keys and NER. Regex alone misses names, so it doesn't count."""
        return bool(self.phi_hmac_key and self.phi_encryption_key and self.phi_ner_enabled)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.disable_auth and self.environment == "prod":
            raise ValueError("DISABLE_AUTH can't be used in prod.")
        if self.phi_hmac_key and len(self.phi_hmac_key) < 32:
            raise ValueError("PHI_HMAC_KEY must be at least 32 characters.")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be below CHUNK_TARGET_TOKENS.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
