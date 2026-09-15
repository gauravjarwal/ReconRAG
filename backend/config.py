from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    google_api_key: str
    chroma_persist_dir: str = "/app/chroma_data"

    # Embedding
    gemini_embedding_model: str = "models/gemini-embedding-001"
    embedding_batch_size: int = 100

    # Generation
    claude_model: str = "claude-haiku-4-5"
    gemini_generation_model: str = "gemini-3.1-flash-lite"

    # Retrieval
    vector_top_k: int = 10
    bm25_top_k: int = 10
    rerank_top_n: int = 3
    confidence_threshold: float = 4.0

    # Security
    max_query_length: int = 2000
    max_file_size_bytes: int = 10 * 1024 * 1024        # 10 MB
    max_request_size_bytes: int = 50 * 1024 * 1024    # 50 MB
    max_files_per_request: int = 20
    max_output_length: int = 4000
    rate_limit_query: str = "30/minute"
    rate_limit_upload: str = "10/minute"


settings = Settings()
