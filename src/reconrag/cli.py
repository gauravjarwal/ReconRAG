"""Command line entry point."""

from __future__ import annotations

import argparse
import base64
import os
import secrets
import time
import uuid
from pathlib import Path

from reconrag.config import get_settings
from reconrag.domain import TERMINAL_STATUSES
from reconrag.errors import NotFound
from reconrag.storage import repo
from reconrag.storage.blobs import BlobStore
from reconrag.storage.db import create_pool
from reconrag.worker import queue

_PLACEHOLDER = ("replace_with", "your_")


def _generated_values() -> dict[str, str]:
    api_key = secrets.token_urlsafe(32)
    return {
        "API_KEYS": api_key,
        "FRONTEND_API_KEY": api_key,
        "POSTGRES_PASSWORD": secrets.token_urlsafe(24),
        "PHI_HMAC_KEY": secrets.token_urlsafe(48),
        "PHI_ENCRYPTION_KEY": base64.urlsafe_b64encode(os.urandom(32)).decode(),
    }


def write_secrets(path: Path) -> list[str]:
    """Fills in missing secrets and keeps existing ones. Returns names, never values."""
    lines = path.read_text("utf-8").splitlines() if path.exists() else []
    present = {}
    for line in lines:
        name, _, value = line.partition("=")
        if value and not value.startswith(_PLACEHOLDER):
            present[name.strip()] = value
    generated = _generated_values()
    if "API_KEYS" in present:
        generated["FRONTEND_API_KEY"] = present["API_KEYS"].split(",")[0]
    added = {k: v for k, v in generated.items() if k not in present}
    kept = [line for line in lines if line.partition("=")[0].strip() not in added]
    path.write_text("\n".join([*kept, *(f"{k}={v}" for k, v in added.items())]) + "\n", "utf-8")
    path.chmod(0o600)
    return sorted(added)


def _migrate() -> None:
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", "migrations")
    command.upgrade(config, "head")


def _ingest(folder: Path, collection_name: str, wait: bool, phi: bool) -> int:
    """Walks a folder and queues every file. Unsupported files are quarantined with a reason."""
    settings = get_settings()
    blobs = BlobStore(settings.blob_dir)
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("."))
    with create_pool(settings, min_size=1) as pool:
        with pool.connection() as conn:
            try:
                collection = repo.get_collection(conn, collection_name)
            except NotFound:
                collection = repo.create_collection(conn, collection_name, phi)
        run_id = uuid.uuid4().hex
        job_ids = []
        for path in paths:
            with path.open("rb") as handle:
                key, size = blobs.put(
                    iter(lambda: handle.read(1 << 20), b""), settings.max_file_bytes
                )
            with pool.connection() as conn:
                job_ids.append(
                    queue.enqueue(
                        conn,
                        collection.id,
                        path.name,
                        key,
                        bytes.fromhex(key),
                        size,
                        request_id=run_id,
                        entry_point="cli",
                    )
                )
        print(f"Queued {len(job_ids)} files in {collection_name}. Run id: {run_id}")
        if not wait:
            return 0
        return _wait(pool, job_ids)


def _wait(pool, job_ids: list) -> int:  # type: ignore[no-untyped-def]
    while True:
        with pool.connection() as conn:
            rows = [queue.get_job(conn, job_id) for job_id in job_ids]
        finished = [r for r in rows if r["status"] in TERMINAL_STATUSES]
        print(f"\r{len(finished)}/{len(rows)} finished", end="", flush=True)
        if len(finished) == len(rows):
            break
        time.sleep(1)
    print()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] in ("quarantined", "dead"):
            print(f"  {row['status']}: {row['filename']}: {row['error_code']}")
    print(counts)
    return 1 if counts.get("dead") else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="reconrag")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="Apply database migrations")
    commands.add_parser("worker", help="Run one ingest worker")
    api = commands.add_parser("api", help="Run the API server")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    secrets_cmd = commands.add_parser("secrets", help="Fill in missing secrets in an env file")
    secrets_cmd.add_argument("--env-file", type=Path, default=Path(".env"))
    ingest = commands.add_parser("ingest", help="Queue every file in a folder")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--collection", required=True)
    ingest.add_argument("--wait", action="store_true", help="Show progress until finished")
    ingest.add_argument("--phi", action="store_true", help="Create the collection as PHI-protected")
    retry = commands.add_parser("retry-job", help="Requeue a dead job")
    retry.add_argument("job_id")
    args = parser.parse_args()

    if args.command == "migrate":
        _migrate()
    elif args.command == "worker":
        from reconrag.worker.main import main as worker_main

        worker_main()
    elif args.command == "api":
        import uvicorn

        uvicorn.run(
            "reconrag.api.app:app_factory",
            factory=True,
            host=args.host,
            port=args.port,
            log_config=None,
        )
    elif args.command == "secrets":
        added = write_secrets(args.env_file)
        print(f"Added: {', '.join(added)}" if added else "All secrets were already set.")
    elif args.command == "ingest":
        raise SystemExit(_ingest(args.folder, args.collection, args.wait, args.phi))
    elif args.command == "retry-job":
        with create_pool(get_settings(), min_size=1) as pool, pool.connection() as conn:
            queue.retry_dead(conn, args.job_id)
