"""Initial schema.

Revision ID: 0001
Revises:
"""

from alembic import op

from reconrag.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dim = get_settings().embedding_dim
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE TABLE collections (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name text NOT NULL UNIQUE,
            phi_protection boolean NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    # A document is the logical thing; each upload that changes it is a version.
    op.execute(
        """
        CREATE TABLE documents (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            collection_id uuid NOT NULL REFERENCES collections ON DELETE CASCADE,
            document_key text NOT NULL,
            title text NOT NULL DEFAULT '',
            retired_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (collection_id, document_key)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE document_versions (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id uuid NOT NULL REFERENCES documents ON DELETE CASCADE,
            collection_id uuid NOT NULL REFERENCES collections ON DELETE CASCADE,
            version integer NOT NULL,
            edition integer NOT NULL DEFAULT 0,
            filename text NOT NULL,
            blob_key text NOT NULL,
            content_hash bytea NOT NULL,
            text_hash bytea NOT NULL,
            simhash bigint NOT NULL,
            near_duplicate_of uuid REFERENCES document_versions ON DELETE SET NULL,
            parser text NOT NULL,
            language text NOT NULL DEFAULT '',
            page_count integer,
            ocr_pages integer NOT NULL DEFAULT 0,
            quality real NOT NULL,
            quality_flags text[] NOT NULL DEFAULT '{}',
            active boolean NOT NULL DEFAULT false,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (document_id, version)
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX one_active_version ON document_versions (document_id) WHERE active"
    )
    op.execute("CREATE INDEX versions_content ON document_versions (collection_id, content_hash)")
    op.execute("CREATE INDEX versions_text ON document_versions (collection_id, text_hash)")
    op.execute(f"""
        CREATE TABLE chunks (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            version_id uuid NOT NULL REFERENCES document_versions ON DELETE CASCADE,
            document_id uuid NOT NULL REFERENCES documents ON DELETE CASCADE,
            collection_id uuid NOT NULL REFERENCES collections ON DELETE CASCADE,
            ordinal integer NOT NULL,
            text text NOT NULL,
            context text NOT NULL DEFAULT '',
            section_path text[] NOT NULL DEFAULT '{{}}',
            page_start integer,
            page_end integer,
            kind text NOT NULL,
            token_count integer NOT NULL,
            text_hash bytea NOT NULL,
            injection_risk boolean NOT NULL DEFAULT false,
            active boolean NOT NULL DEFAULT false,
            embedding halfvec({dim}) NOT NULL,
            search tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', context), 'A') ||
                setweight(to_tsvector('simple', context), 'A') ||
                setweight(to_tsvector('english', text), 'C') ||
                setweight(to_tsvector('simple', text), 'C')
            ) STORED,
            UNIQUE (version_id, ordinal)
        )
    """)
    # Partial indexes: superseded chunks stay stored for audit but cost nothing at query time.
    op.execute(
        """
        CREATE INDEX chunks_embedding ON chunks
        USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)
        WHERE active
        """
    )
    op.execute("CREATE INDEX chunks_search ON chunks USING gin (search) WHERE active")
    op.execute("CREATE INDEX chunks_trgm ON chunks USING gin (text gin_trgm_ops) WHERE active")
    op.execute("CREATE INDEX chunks_collection ON chunks (collection_id) WHERE active")
    op.execute("CREATE INDEX chunks_reuse ON chunks (collection_id, text_hash)")
    op.execute(
        """
        CREATE TABLE ingest_jobs (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            collection_id uuid NOT NULL REFERENCES collections ON DELETE CASCADE,
            filename text NOT NULL,
            blob_key text NOT NULL,
            content_hash bytea NOT NULL,
            size_bytes bigint NOT NULL,
            document_key text,
            status text NOT NULL DEFAULT 'queued',
            stage text NOT NULL DEFAULT 'queued',
            attempts integer NOT NULL DEFAULT 0,
            worker_id text,
            lease_expires_at timestamptz,
            not_before timestamptz NOT NULL DEFAULT now(),
            error_code text,
            error_detail text,
            version_id uuid REFERENCES document_versions ON DELETE SET NULL,
            stats jsonb NOT NULL DEFAULT '{}',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz
        )
        """
    )
    op.execute(
        """
        CREATE INDEX jobs_claimable ON ingest_jobs (not_before, created_at)
        WHERE status IN ('queued', 'running')
        """
    )
    op.execute("CREATE INDEX jobs_by_collection ON ingest_jobs (collection_id, created_at DESC)")
    op.execute(
        """
        CREATE TABLE phi_tokens (
            collection_id uuid NOT NULL REFERENCES collections ON DELETE CASCADE,
            token text NOT NULL,
            entity_type text NOT NULL,
            ciphertext bytea NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (collection_id, token)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE phi_aliases (
            collection_id uuid NOT NULL REFERENCES collections ON DELETE CASCADE,
            alias_token text NOT NULL,
            canonical_token text,
            PRIMARY KEY (collection_id, alias_token)
        )
        """
    )


def downgrade() -> None:
    for table in (
        "phi_aliases",
        "phi_tokens",
        "ingest_jobs",
        "chunks",
        "document_versions",
        "documents",
        "collections",
    ):
        op.execute(f"DROP TABLE {table}")
