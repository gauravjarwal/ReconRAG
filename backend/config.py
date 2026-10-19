from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> frozenset[str]:
    return frozenset(v.strip() for v in value.split(",") if v.strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Optional: empty means Gemini-only generation (logged at startup).
    anthropic_api_key: str = ""
    google_api_key: str
    chroma_persist_dir: str = "/app/chroma_data"

    # Embedding
    gemini_embedding_model: str = "models/gemini-embedding-001"
    embedding_batch_size: int = 100

    # Generation
    claude_model: str = "claude-haiku-4-5"
    gemini_generation_model: str = "gemini-3.1-flash-lite"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

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
    rate_limit_default: str = "200/minute"
    cors_origins: str = "http://localhost:8501,http://frontend:8501"

    # Authentication: comma-separated API keys sent in the X-API-Key header.
    # API_KEYS can read/write collections but only ever see pseudonymized PHI.
    # PHI_ACCESS_API_KEYS can additionally request re-identified answers.
    api_keys: str = ""
    phi_access_api_keys: str = ""

    # PHI protection
    phi_hmac_key: str                     # keys pseudonym tokens; >= 32 chars
    phi_encryption_key: str               # Fernet key encrypting the token vault
    phi_vault_path: str = "/app/phi_vault/vault.db"
    phi_ner_enabled: bool = True          # False = regex-only detection (misses names)
    phi_ner_model: str = "en_core_web_lg"

    @property
    def api_key_set(self) -> frozenset[str]:
        return _split_csv(self.api_keys)

    @property
    def phi_access_key_set(self) -> frozenset[str]:
        return _split_csv(self.phi_access_api_keys)

    @property
    def cors_origin_list(self) -> list[str]:
        return sorted(_split_csv(self.cors_origins))

    @model_validator(mode="after")
    def _check_secrets(self) -> Settings:
        if not (self.api_key_set or self.phi_access_key_set):
            raise ValueError("Set API_KEYS and/or PHI_ACCESS_API_KEYS: the API refuses to run unauthenticated.")
        if len(self.phi_hmac_key) < 32:
            raise ValueError("PHI_HMAC_KEY must be at least 32 characters.")
        return self


settings = Settings()
