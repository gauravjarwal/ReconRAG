from __future__ import annotations

import sqlite3

import pytest
from cryptography.fernet import Fernet

from config import settings
from conftest import spacy_model_available
from pipelines import phi
from pipelines.phi_vault import PhiVault, VaultIntegrityError
from services.deidentification import (
    TOKEN_PATTERN,
    make_token,
    pseudonymize_document,
    pseudonymize_query,
    reidentify,
)

NOTE = (
    "Patient: John Smith\n"
    "MRN: 00984321  DOB: 03/14/1952  Phone: (217) 555-0143\n"
    "Mr. Smith was admitted on Jan 5, 2024 with chest pain."
)
RAW_VALUES = ["John Smith", "00984321", "03/14/1952", "(217) 555-0143", "Jan 5, 2024"]


@pytest.fixture(scope="module", autouse=True)
def _ner():
    if spacy_model_available():
        phi.load_ner("en_core_web_lg")


def test_tokens_are_deterministic_keyed_and_collection_scoped():
    a = make_token("ward-a", "SSN", "123-45-6789")
    assert a == make_token("ward-a", "SSN", "123 45 6789")  # normalized
    assert a != make_token("ward-b", "SSN", "123-45-6789")  # no cross-collection linkage
    assert TOKEN_PATTERN.fullmatch(a)
    assert make_token("ward-a", "PERSON", "John Smith") == make_token("ward-a", "PERSON", "JOHN  smith")


def test_document_pseudonymization_removes_all_raw_values_and_round_trips():
    result = pseudonymize_document("deid-roundtrip", NOTE)
    for value in RAW_VALUES:
        assert value not in result.text
    assert result.entity_counts["MRN"] == 1
    # "Mr. Smith" is linked to John Smith, so it re-identifies to the full name.
    assert reidentify("deid-roundtrip", result.text) == NOTE.replace("Mr. Smith", "Mr. John Smith")


@pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_lg not installed")
def test_partial_name_shares_full_name_token_and_query_alias_resolves():
    doc = pseudonymize_document("deid-alias", NOTE).text
    full = make_token("deid-alias", "PERSON", "John Smith")
    assert doc.count(full) == 2  # "Patient: John Smith" and "Mr. Smith"
    assert make_token("deid-alias", "PERSON", "Smith") not in doc

    query = pseudonymize_query("deid-alias", "When was Smith admitted?")
    assert full in query.text and "Smith" not in query.text


@pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_lg not installed")
def test_ambiguous_surname_is_not_linked():
    text = "John Smith and Jane Smith are siblings. Smith was discharged."
    spans = [s for s in phi.detect_phi(text) if s.text == "Smith"]
    assert spans and all(s.canonical is None for s in spans)


def test_alias_seen_with_two_canonicals_becomes_ambiguous(tmp_path):
    vault = PhiVault(str(tmp_path / "v.db"), Fernet.generate_key().decode())
    vault.store_aliases("c", {"[PERSON_aaaaaaaaaaaa]": "[PERSON_111111111111]"})
    assert vault.resolve_aliases("c", ["[PERSON_aaaaaaaaaaaa]"]) == {"[PERSON_aaaaaaaaaaaa]": "[PERSON_111111111111]"}
    vault.store_aliases("c", {"[PERSON_aaaaaaaaaaaa]": "[PERSON_222222222222]"})
    assert vault.resolve_aliases("c", ["[PERSON_aaaaaaaaaaaa]"]) == {}
    vault.store_aliases("c", {"[PERSON_aaaaaaaaaaaa]": "[PERSON_111111111111]"})
    assert vault.resolve_aliases("c", ["[PERSON_aaaaaaaaaaaa]"]) == {}  # stays ambiguous


def test_reidentify_is_scoped_to_collection():
    text = pseudonymize_document("deid-scope-a", "MRN: 55667788").text
    assert reidentify("deid-scope-b", text) == text


def test_reidentify_tolerates_uppercased_hex_from_llm():
    text = pseudonymize_document("deid-case", "MRN: 11223344").text
    assert reidentify("deid-case", text.upper()) == "MRN: 11223344"


@pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_lg not installed")
def test_query_uses_vault_to_catch_names_ner_misses():
    doc = pseudonymize_document("deid-query", NOTE).text
    person_token = make_token("deid-query", "PERSON", "John Smith")
    assert person_token in doc

    query = pseudonymize_query("deid-query", "what meds is john smith on")
    assert "john smith" not in query.text
    assert person_token in query.text


def test_vault_encrypts_values_at_rest(tmp_path):
    path = str(tmp_path / "v.db")
    vault = PhiVault(path, Fernet.generate_key().decode())
    vault.store("c", {"[SSN_aaaaaaaaaaaa]": ("SSN", "123-45-6789")})
    raw = sqlite3.connect(path).execute("SELECT ciphertext FROM phi_tokens").fetchone()[0]
    assert b"123-45-6789" not in raw
    assert vault.resolve("c", ["[SSN_aaaaaaaaaaaa]"]) == {"[SSN_aaaaaaaaaaaa]": "123-45-6789"}


def test_vault_detects_swapped_rows(tmp_path):
    path = str(tmp_path / "v.db")
    vault = PhiVault(path, Fernet.generate_key().decode())
    vault.store("c", {"[SSN_aaaaaaaaaaaa]": ("SSN", "111-11-1111"), "[SSN_bbbbbbbbbbbb]": ("SSN", "222-22-2222")})
    conn = sqlite3.connect(path)
    with conn:
        a = conn.execute("SELECT ciphertext FROM phi_tokens WHERE token='[SSN_aaaaaaaaaaaa]'").fetchone()[0]
        conn.execute("UPDATE phi_tokens SET ciphertext=? WHERE token='[SSN_bbbbbbbbbbbb]'", (a,))
    with pytest.raises(VaultIntegrityError):
        vault.resolve("c", ["[SSN_bbbbbbbbbbbb]"])


def test_vault_rejects_wrong_key(tmp_path):
    path = str(tmp_path / "v.db")
    PhiVault(path, Fernet.generate_key().decode()).store("c", {"[SSN_aaaaaaaaaaaa]": ("SSN", "1")})
    with pytest.raises(VaultIntegrityError):
        PhiVault(path, Fernet.generate_key().decode()).resolve("c", ["[SSN_aaaaaaaaaaaa]"])


def test_settings_require_strong_hmac_key(monkeypatch):
    from config import Settings

    monkeypatch.setenv("PHI_HMAC_KEY", "short")
    with pytest.raises(ValueError, match="PHI_HMAC_KEY"):
        Settings()
    assert len(settings.phi_hmac_key) >= 32
