from __future__ import annotations

import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_HEADERS = {"X-API-Key": os.environ.get("BACKEND_API_KEY", "")}


class ApiError(Exception):
    """Backend returned an error. `detail` is the backend's safe-to-show message."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _url(path: str) -> str:
    return f"{BACKEND_URL}{path}"


def _check(resp: requests.Response):
    if resp.ok:
        return resp.json()
    try:
        detail = resp.json().get("detail", resp.reason)
    except ValueError:
        detail = resp.reason
    raise ApiError(resp.status_code, detail if isinstance(detail, str) else str(detail))


def list_collections() -> list[dict]:
    return _check(requests.get(_url("/api/collections"), headers=_HEADERS, timeout=10))


def create_collection(name: str, phi_protection: bool = True) -> dict:
    return _check(requests.post(
        _url("/api/collections"),
        json={"name": name, "phi_protection": phi_protection},
        headers=_HEADERS,
        timeout=10,
    ))


def upload_documents(collection_name: str, files: list[tuple[str, bytes, str]]) -> dict:
    """
    Args:
        files: list of (filename, file_bytes, content_type)
    """
    multipart = [("files", (name, data, ct)) for name, data, ct in files]
    return _check(requests.post(
        _url(f"/api/collections/{collection_name}/documents"),
        files=multipart,
        headers=_HEADERS,
        timeout=300,
    ))


def list_documents(collection_name: str) -> list[str]:
    return _check(requests.get(_url(f"/api/collections/{collection_name}/documents"), headers=_HEADERS, timeout=10))


def query_collection(collection_name: str, question: str, reranking: bool = True, reidentify: bool = False) -> dict:
    return _check(requests.post(
        _url(f"/api/collections/{collection_name}/query"),
        json={"question": question, "reranking": reranking, "reidentify": reidentify},
        headers=_HEADERS,
        timeout=120,
    ))


def health_check() -> bool:
    try:
        resp = requests.get(_url("/api/health"), timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
