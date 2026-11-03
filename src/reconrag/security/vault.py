"""Encrypted token-to-PHI store, in Postgres so every replica shares it.

Values are Fernet-encrypted with a key held outside the database, so a dump alone reveals nothing.
The token is encrypted with its value and checked on decrypt, so swapped rows are detected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken

from reconrag.storage.db import Conn


class VaultIntegrityError(Exception):
    pass


class PhiVault:
    def __init__(self, encryption_key: str) -> None:
        self._fernet = Fernet(encryption_key.encode())

    def store(
        self, conn: Conn, collection_id: UUID, entries: Mapping[str, tuple[str, str]]
    ) -> None:
        """entries: token -> (entity_type, original value). Existing tokens are left alone."""
        rows = [
            (collection_id, token, kind, self._fernet.encrypt(f"{token}\x00{value}".encode()))
            for token, (kind, value) in entries.items()
        ]
        if not rows:
            return
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO phi_tokens (collection_id, token, entity_type, ciphertext)
                VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
                """,
                rows,
            )

    def store_aliases(self, conn: Conn, collection_id: UUID, aliases: Mapping[str, str]) -> None:
        """An alias seen with two different full names becomes ambiguous (NULL), not a guess."""
        if not aliases:
            return
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO phi_aliases AS a (collection_id, alias_token, canonical_token)
                VALUES (%s, %s, %s)
                ON CONFLICT (collection_id, alias_token) DO UPDATE SET canonical_token =
                    CASE WHEN a.canonical_token = EXCLUDED.canonical_token
                         THEN a.canonical_token END
                """,
                [(collection_id, alias, canonical) for alias, canonical in aliases.items()],
            )

    def known(self, conn: Conn, collection_id: UUID, tokens: Sequence[str]) -> set[str]:
        rows = conn.execute(
            "SELECT token FROM phi_tokens WHERE collection_id = %s AND token = ANY(%s)",
            (collection_id, list(tokens)),
        ).fetchall()
        return {row["token"] for row in rows} | set(self.aliases(conn, collection_id, tokens))

    def aliases(self, conn: Conn, collection_id: UUID, tokens: Sequence[str]) -> dict[str, str]:
        rows = conn.execute(
            """
            SELECT alias_token, canonical_token FROM phi_aliases
            WHERE collection_id = %s AND alias_token = ANY(%s) AND canonical_token IS NOT NULL
            """,
            (collection_id, list(tokens)),
        ).fetchall()
        return {row["alias_token"]: row["canonical_token"] for row in rows}

    def resolve(self, conn: Conn, collection_id: UUID, tokens: Sequence[str]) -> dict[str, str]:
        rows = conn.execute(
            "SELECT token, ciphertext FROM phi_tokens WHERE collection_id = %s AND token = ANY(%s)",
            (collection_id, list(tokens)),
        ).fetchall()
        resolved: dict[str, str] = {}
        for row in rows:
            try:
                plain = self._fernet.decrypt(bytes(row["ciphertext"])).decode()
            except InvalidToken as exc:
                raise VaultIntegrityError("Wrong key or a tampered row.") from exc
            token, _, value = plain.partition("\x00")
            if token != row["token"]:
                raise VaultIntegrityError("A row was bound to a different token.")
            resolved[token] = value
        return resolved
