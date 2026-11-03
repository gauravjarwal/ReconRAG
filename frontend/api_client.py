"""A small client for the v2 API. Errors become ApiError with a message safe to show."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_HEADERS = {"X-API-Key": os.environ.get("BACKEND_API_KEY", "")}
_TIMEOUT = httpx.Timeout(30.0, read=120.0)


class ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _check(response: httpx.Response) -> Any:
    if response.is_success:
        return response.json() if response.content else None
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError):
        message = response.reason_phrase
    raise ApiError(response.status_code, message)


def _client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND_URL, headers=_HEADERS, timeout=_TIMEOUT)


def is_ready() -> bool:
    try:
        with _client() as client:
            return client.get("/readyz").status_code == 200
    except httpx.HTTPError:
        return False


def list_collections() -> list[dict[str, Any]]:
    with _client() as client:
        return _check(client.get("/v1/collections"))


def create_collection(name: str, phi_protection: bool) -> None:
    with _client() as client:
        _check(
            client.post("/v1/collections", json={"name": name, "phi_protection": phi_protection})
        )


def upload(collection: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
    with _client() as client:
        return _check(
            client.post(
                f"/v1/collections/{collection}/documents",
                files=[("files", file) for file in files],
            )
        )


def jobs(collection: str, limit: int = 50) -> list[dict[str, Any]]:
    with _client() as client:
        response = client.get(f"/v1/collections/{collection}/jobs", params={"limit": limit})
        return _check(response)["data"]


def documents(collection: str, limit: int = 200) -> list[dict[str, Any]]:
    with _client() as client:
        response = client.get(f"/v1/collections/{collection}/documents", params={"limit": limit})
        return _check(response)["data"]


def chunk(collection: str, chunk_id: str) -> dict[str, Any]:
    with _client() as client:
        return _check(client.get(f"/v1/collections/{collection}/chunks/{chunk_id}"))


def ask_stream(
    collection: str, question: str, reidentify: bool
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yields (event, data) as they arrive."""
    body = {"question": question, "reidentify": reidentify}
    with (
        _client() as client,
        client.stream("POST", f"/v1/collections/{collection}/query/stream", json=body) as response,
    ):
        if not response.is_success:
            response.read()
            _check(response)
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                yield event, json.loads(line.removeprefix("data: "))
