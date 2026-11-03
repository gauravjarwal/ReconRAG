"""Runs parsing in a child process, so a hostile file can only hurt the child.

The child outlives jobs to keep OCR models loaded. Kept apart from ProcessPoolExecutor because
killing one stuck task there needs private attributes.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any

from reconrag.config import Settings
from reconrag.errors import RejectedFile
from reconrag.ingestion.prepare import Prepared, prepare

_CONTEXT = mp.get_context("spawn")  # Fork is unsafe here: the parent has heartbeat threads.


class ParseFailed(Exception):
    """The child died or timed out. Possibly the file's fault, so the job is retried a few times."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _serve(pipe: Connection, settings: Settings) -> None:
    while True:
        try:
            data, filename = pipe.recv()
        except EOFError:
            return
        # Exceptions with custom constructors don't survive pickling, so errors travel as tuples.
        reply: tuple[Any, ...]
        try:
            reply = ("ok", prepare(data, filename, settings))
        except RejectedFile as exc:
            reply = ("rejected", exc.code, str(exc))
        except Exception as exc:  # A parser bug must not end the child's service loop.
            reply = ("error", type(exc).__name__, str(exc)[:500])
        pipe.send(reply)


class ParseSandbox:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._process: Any = None
        self._pipe: Connection | None = None

    def _start(self) -> None:
        parent, child = _CONTEXT.Pipe()
        self._process = _CONTEXT.Process(
            target=_serve, args=(child, self._settings), daemon=True, name="reconrag-parser"
        )
        self._process.start()
        child.close()
        self._pipe = parent

    def _alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def stop(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.join(5)
        if self._pipe is not None:
            self._pipe.close()
        self._process = None
        self._pipe = None

    def prepare(self, data: bytes, filename: str, timeout_s: float) -> Prepared:
        if not self._alive():
            self.stop()
            self._start()
        assert self._pipe is not None
        try:
            self._pipe.send((data, filename))
            if not self._pipe.poll(timeout_s):
                self.stop()
                raise ParseFailed("parse_timeout", f"Parsing took over {timeout_s:.0f} seconds.")
            reply = self._pipe.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            self.stop()
            raise ParseFailed("parser_crashed", "The parser process died.") from exc
        if reply[0] == "ok":
            return reply[1]
        if reply[0] == "rejected":
            raise RejectedFile(reply[1], reply[2])
        raise ParseFailed("parser_error", f"{reply[1]}: {reply[2]}")
