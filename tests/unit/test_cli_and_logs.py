import json
import logging
from pathlib import Path

from reconrag.cli import write_secrets
from reconrag.observability.logs import JsonFormatter


def test_secrets_are_generated_once_and_kept(tmp_path: Path):
    env = tmp_path / ".env"
    assert "API_KEYS" in write_secrets(env)
    first = env.read_text()
    assert write_secrets(env) == []
    assert env.read_text() == first
    assert oct(env.stat().st_mode)[-3:] == "600"


def test_existing_values_survive_and_placeholders_are_replaced(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=real\nAPI_KEYS=replace_with_generated_key\n")
    write_secrets(env)
    text = env.read_text()
    assert "GOOGLE_API_KEY=real" in text
    assert "replace_with" not in text
    assert sum(line.startswith("API_KEYS=") for line in text.splitlines()) == 1


def test_frontend_key_follows_existing_api_key(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("API_KEYS=abc,def\n")
    write_secrets(env)
    assert "FRONTEND_API_KEY=abc" in env.read_text()


def test_json_logs_redact_structured_phi():
    record = logging.LogRecord("x", logging.INFO, "f", 1, "ssn 123-45-6789", None, None)
    record.note = "mail a@b.com"
    entry = json.loads(JsonFormatter().format(record))
    assert "123-45-6789" not in entry["message"]
    assert "a@b.com" not in entry["note"]
