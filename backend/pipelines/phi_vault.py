"""
Encrypted store mapping pseudonym tokens back to the original PHI values.

This is the only place original PHI is persisted. Values are Fernet-encrypted
(AES-128-CBC + HMAC-SHA256) with a key that lives outside the database, so a
copy of vault.db alone reveals nothing. The token is encrypted together with
the value and checked on decrypt, so swapping ciphertexts between rows is
detected rather than silently re-identifying the wrong person.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_SQLITE_MAX_PARAMS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS phi_tokens (
    collection  TEXT NOT NULL,
    token       TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    ciphertext  BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (collection, token)
);
-- Partial-name token -> full-name token ("Gonzalez" -> "Maria Gonzalez"). Tokens only,
-- no PHI. canonical_token is NULL when the alias is ambiguous across documents.
CREATE TABLE IF NOT EXISTS phi_aliases (
    collection      TEXT NOT NULL,
    alias_token     TEXT NOT NULL,
    canonical_token TEXT,
    PRIMARY KEY (collection, alias_token)
);
"""


class VaultIntegrityError(Exception):
    pass


class PhiVault:
    def __init__(self, path: str, encryption_key: str) -> None:
        self._path = path
        self._fernet = Fernet(encryption_key.encode())
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, mode=0o700, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One connection per call: sqlite3 connections are not shareable across
        # the threadpool FastAPI runs sync endpoints in.
        return sqlite3.connect(self._path, timeout=10)

    def store(self, collection: str, entries: dict[str, tuple[str, str]]) -> None:
        """entries: token -> (entity_type, original_value). Existing tokens are left untouched."""
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (collection, token, entity_type, self._fernet.encrypt(f"{token}\x00{value}".encode()), now)
            for token, (entity_type, value) in entries.items()
        ]
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT OR IGNORE INTO phi_tokens (collection, token, entity_type, ciphertext, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def store_aliases(self, collection: str, aliases: dict[str, str]) -> None:
        """aliases: alias token -> canonical token. An alias seen with two different canonicals becomes ambiguous (NULL)."""
        if not aliases:
            return
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO phi_aliases (collection, alias_token, canonical_token) VALUES (?, ?, ?) "
                "ON CONFLICT (collection, alias_token) DO UPDATE SET canonical_token = "
                "CASE WHEN canonical_token = excluded.canonical_token THEN canonical_token ELSE NULL END",
                [(collection, alias, canonical) for alias, canonical in aliases.items()],
            )

    def _select(self, collection: str, tokens: list[str], sql: str) -> list[tuple]:
        """Run `sql` (with {placeholders} for the token list) in batches under SQLite's parameter limit."""
        rows: list[tuple] = []
        with closing(self._connect()) as conn:
            for i in range(0, len(tokens), _SQLITE_MAX_PARAMS):
                batch = tokens[i : i + _SQLITE_MAX_PARAMS]
                placeholders = ",".join("?" * len(batch))
                rows.extend(conn.execute(sql.format(placeholders=placeholders), [collection, *batch]).fetchall())
        return rows

    def existing(self, collection: str, tokens: list[str]) -> set[str]:
        """Which of these tokens are known in the collection, as values or unambiguous aliases (no decryption)."""
        unique = list(set(tokens))
        known = {row[0] for row in self._select(
            collection, unique, "SELECT token FROM phi_tokens WHERE collection = ? AND token IN ({placeholders})"
        )}
        return known | set(self.resolve_aliases(collection, unique))

    def resolve_aliases(self, collection: str, tokens: list[str]) -> dict[str, str]:
        """alias token -> canonical token, for unambiguous aliases among `tokens`."""
        rows = self._select(
            collection,
            list(set(tokens)),
            "SELECT alias_token, canonical_token FROM phi_aliases "
            "WHERE collection = ? AND alias_token IN ({placeholders}) AND canonical_token IS NOT NULL",
        )
        return dict(rows)

    def resolve(self, collection: str, tokens: list[str]) -> dict[str, str]:
        """token -> original value, for tokens present in the vault."""
        resolved: dict[str, str] = {}
        rows = self._select(
            collection,
            list(set(tokens)),
            "SELECT token, ciphertext FROM phi_tokens WHERE collection = ? AND token IN ({placeholders})",
        )
        for token, ciphertext in rows:
            try:
                stored_token, _, value = self._fernet.decrypt(ciphertext).decode().partition("\x00")
            except InvalidToken as exc:
                raise VaultIntegrityError("Vault entry failed authentication; wrong key or tampered row.") from exc
            if stored_token != token:
                raise VaultIntegrityError("Vault entry is bound to a different token; row was tampered with.")
            resolved[token] = value
        return resolved

    def delete_collection(self, collection: str) -> int:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM phi_aliases WHERE collection = ?", (collection,))
            return conn.execute("DELETE FROM phi_tokens WHERE collection = ?", (collection,)).rowcount
