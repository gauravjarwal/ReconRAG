"""Content-addressed file store. Identical uploads share one file, and writes are atomic."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from reconrag.errors import InvalidInput


class BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise InvalidInput("Bad blob key.")
        return self._root / key[:2] / key[2:4] / key

    def put(self, chunks: Iterable[bytes], max_bytes: int) -> tuple[str, int]:
        """Streams to a temp file while hashing, so the size limit needs no memory."""
        self._root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        handle, temp_name = tempfile.mkstemp(dir=self._root)
        temp = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as out:
                for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise InvalidInput(f"File exceeds the {max_bytes} byte limit.")
                    digest.update(chunk)
                    out.write(chunk)
            key = digest.hexdigest()
            target = self._path(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp.replace(target)
            return key, size
        finally:
            temp.unlink(missing_ok=True)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        """`_path` accepts only a hex digest, so a key can't name anything outside the store."""
        self._path(key).unlink(missing_ok=True)
