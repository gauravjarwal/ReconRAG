from __future__ import annotations

import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _url(path: str) -> str:
    return f"{BACKEND_URL}{path}"


def list_collections() -> list[dict]:
    resp = requests.get(_url("/api/collections"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def create_collection(name: str) -> dict:
    resp = requests.post(_url("/api/collections"), json={"name": name}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def upload_documents(collection_name: str, files: list[tuple[str, bytes, str]]) -> dict:
    """
    Args:
        files: list of (filename, file_bytes, content_type)
    """
    multipart = [("files", (name, data, ct)) for name, data, ct in files]
    resp = requests.post(
        _url(f"/api/collections/{collection_name}/documents"),
        files=multipart,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def list_documents(collection_name: str) -> list[str]:
    resp = requests.get(_url(f"/api/collections/{collection_name}/documents"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def query_collection(collection_name: str, question: str, reranking: bool = True) -> dict:
    resp = requests.post(
        _url(f"/api/collections/{collection_name}/query"),
        json={"question": question, "reranking": reranking},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def health_check() -> bool:
    try:
        resp = requests.get(_url("/api/health"), timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
