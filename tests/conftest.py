from __future__ import annotations

import os
import sys
import tempfile

from cryptography.fernet import Fernet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Settings are read at import time, so the test environment must exist first.
_tmp = tempfile.mkdtemp(prefix="reconrag-tests-")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("API_KEYS", "standard-test-key")
os.environ.setdefault("PHI_ACCESS_API_KEYS", "phi-test-key")
os.environ.setdefault("PHI_HMAC_KEY", "x" * 48)
os.environ.setdefault("PHI_ENCRYPTION_KEY", Fernet.generate_key().decode())
os.environ.setdefault("PHI_VAULT_PATH", os.path.join(_tmp, "vault.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(_tmp, "chroma"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def spacy_model_available() -> bool:
    try:
        import spacy.util

        return spacy.util.is_package("en_core_web_lg")
    except ImportError:
        return False
