from __future__ import annotations

from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from reconrag.domain import Block, BlockKind, Collection, Hit, ParsedDocument
from reconrag.generation.service import Answerer
from reconrag.llm.base import Usage
from reconrag.llm.breaker import CircuitBreaker
from reconrag.llm.router import LLMRouter
from reconrag.retrieval.context import Passage
from reconrag.retrieval.service import Retrieved
from reconrag.security.phi import TOKEN_PATTERN, Pseudonymizer
from reconrag.security.vault import PhiVault, VaultIntegrityError
from reconrag.storage import repo
from reconrag.storage.db import Pool
from reconrag.worker import queue
from reconrag.worker.processor import Processor

NOTE = (
    "Patient: Maria Gonzalez was seen by Dr. Alan Reed. SSN 123-45-6789. "
    "MRN: AB12345. Ms. Gonzalez reports dizziness. Follow up in clinic."
)


@pytest.fixture
def collection(pool: Pool) -> Collection:
    with pool.connection() as conn:
        return repo.create_collection(conn, "ward", phi_protection=True)


@pytest.fixture
def pseudonymizer() -> Pseudonymizer:
    return Pseudonymizer("k" * 48, PhiVault(Fernet.generate_key().decode()))


def document(*blocks: Block) -> ParsedDocument:
    return ParsedDocument(list(blocks), parser="t")


def test_identifiers_leave_the_text_and_can_come_back(pool, collection, pseudonymizer):
    with pool.connection() as conn:
        result, counts = pseudonymizer.document(
            conn, collection.id, document(Block(BlockKind.TEXT, NOTE))
        )
        text = result.blocks[0].text
        restored = pseudonymizer.reidentify(conn, collection.id, text)
    assert "123-45-6789" not in text
    assert "AB12345" not in text
    assert "Gonzalez" not in text
    assert counts["SSN"] == 1
    assert "123-45-6789" in restored
    assert "Maria Gonzalez" in restored


def test_table_cells_are_protected_and_keep_their_shape(pool, collection, pseudonymizer):
    table = Block(BlockKind.TABLE, rows=[["Name", "SSN"], ["Patient: Ana Ruiz", "321-54-9876"]])
    with pool.connection() as conn:
        result, _ = pseudonymizer.document(conn, collection.id, document(table))
    rows = result.blocks[0].rows
    assert rows is not None
    assert len(rows[1]) == 2
    assert "321-54-9876" not in str(rows)
    assert "Ruiz" not in str(rows)


def test_same_person_gets_the_same_token_in_a_query(pool, collection, pseudonymizer):
    with pool.connection() as conn:
        result, _ = pseudonymizer.document(
            conn, collection.id, document(Block(BlockKind.TEXT, NOTE))
        )
        query = pseudonymizer.query(conn, collection.id, "What does maria gonzalez report?")
    token = TOKEN_PATTERN.search(query.text)
    assert token is not None
    assert token.group(0) in result.blocks[0].text
    assert "gonzalez" not in query.text


def test_tokens_differ_between_collections(pool, pseudonymizer, collection):
    with pool.connection() as conn:
        other = repo.create_collection(conn, "other-ward", phi_protection=True)
    first = pseudonymizer.token(collection.id, "SSN", "123-45-6789")
    assert first != pseudonymizer.token(other.id, "SSN", "123-45-6789")


def test_vault_holds_no_plaintext(pool, collection, pseudonymizer):
    with pool.connection() as conn:
        pseudonymizer.document(conn, collection.id, document(Block(BlockKind.TEXT, NOTE)))
        rows = conn.execute("SELECT ciphertext FROM phi_tokens").fetchall()
    assert rows
    assert all(b"123-45-6789" not in bytes(row["ciphertext"]) for row in rows)


def test_swapped_vault_rows_are_detected(pool, collection, pseudonymizer):
    with pool.connection() as conn:
        pseudonymizer.document(conn, collection.id, document(Block(BlockKind.TEXT, NOTE)))
        tokens = [r["token"] for r in conn.execute("SELECT token FROM phi_tokens").fetchall()]
        conn.execute(
            """
            UPDATE phi_tokens SET ciphertext =
                (SELECT ciphertext FROM phi_tokens WHERE token = %s) WHERE token = %s
            """,
            (tokens[0], tokens[1]),
        )
        with pytest.raises(VaultIntegrityError):
            pseudonymizer.vault.resolve(conn, collection.id, tokens[:2])


def test_nothing_identifying_reaches_the_embedder(
    pool, blobs, parser, embedder, settings, collection, pseudonymizer
):
    key, size = blobs.put([NOTE.encode() * 3], 10_000)
    with pool.connection() as conn:
        queue.enqueue(conn, collection.id, "maria_gonzalez.txt", key, bytes.fromhex(key), size)
        job = queue.claim(conn, "w", 30, 3)
    assert job is not None
    Processor(pool, blobs, parser, embedder, settings, "w", pseudonymizer).run(job)
    sent = " ".join(embedder.embedded)
    assert sent
    for secret in ("123-45-6789", "AB12345", "Gonzalez", "gonzalez", "maria"):
        assert secret not in sent


def test_a_phi_collection_never_streams_unvalidated_tokens(
    pool, collection, pseudonymizer, settings
):
    hit = Hit(
        chunk_id=uuid4(),
        version_id=uuid4(),
        document_id=uuid4(),
        filename="note.txt",
        ordinal=0,
        text="The dose is 40 mg.",
        context="",
        section_path=[],
        page_start=None,
        page_end=None,
        kind="text",
        text_hash=b"x",
        injection_risk=False,
        rerank_score=5.0,
    )
    retrieved = Retrieved(
        [Passage("S1", hit, hit.text)], [hit], refused=False, reason="ok", confidence=5.0
    )

    class Stub:
        def retrieve(self, *_args):
            return retrieved

    class Model:
        name = "stub"
        model = "gemini-3.1-flash-lite"

        def stream(self, system, user, max_tokens):
            yield "The dose is 40 mg [S1]."
            yield Usage(10, 5)

    router = LLMRouter([Model()], {"stub": CircuitBreaker(3, 30)})
    answerer = Answerer(pool, Stub(), router, settings, pseudonymizer)
    kinds = [event.kind for event in answerer.stream(collection, "What is the dose?", False)]
    assert kinds == ["sources", "done"]
