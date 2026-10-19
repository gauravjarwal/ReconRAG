from __future__ import annotations

import pathlib

import pytest

from conftest import spacy_model_available
from pipelines import phi
from pipelines.phi import PhiSpan, detect_phi, detect_structured_phi, redact_structured_phi, resolve_overlaps

SAMPLES = pathlib.Path(__file__).parent.parent / "samples"


def _found(text: str) -> set[tuple[str, str]]:
    return {(s.entity_type, s.text) for s in detect_structured_phi(text)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SSN 123-45-6789 on file", ("SSN", "123-45-6789")),
        ("Social Security No: 123456789", ("SSN", "123456789")),
        ("MRN: A1234567", ("MRN", "A1234567")),
        ("Medical record number 00984321", ("MRN", "00984321")),
        ("Member ID: XYZ123456789", ("HEALTH_PLAN_ID", "XYZ123456789")),
        ("MBI 1EG4TE5MK73 active", ("HEALTH_PLAN_ID", "1EG4TE5MK73")),
        ("Acct #: 55512-889", ("ACCOUNT_NUMBER", "55512-889")),
        ("DEA number AB1234563", ("LICENSE_NUMBER", "AB1234563")),
        ("Pacemaker serial no. PM-88231-X", ("DEVICE_ID", "PM-88231-X")),
        ("VIN 1HGCM82633A004352", ("VEHICLE_ID", "1HGCM82633A004352")),
        ("Card 4111 1111 1111 1111 declined", ("CREDIT_CARD", "4111 1111 1111 1111")),
        ("Call (217) 555-0143 after 5", ("PHONE", "(217) 555-0143")),
        ("Fax: 2175550199", ("PHONE", "2175550199")),
        ("Email jane.doe@example.org today", ("EMAIL", "jane.doe@example.org")),
        ("Portal at https://portal.example.org/u/8812.", ("URL", "https://portal.example.org/u/8812")),
        ("Login from 10.20.30.40", ("IP_ADDRESS", "10.20.30.40")),
        ("Lives at 742 Evergreen Terrace", ("ADDRESS", "742 Evergreen Terrace")),
        ("Springfield, IL 62704", ("ZIP", "62704")),
        ("DOB 03/14/1952", ("DATE", "03/14/1952")),
        ("Admitted 2024-01-05", ("DATE", "2024-01-05")),
        ("Seen on Jan 5, 2024 for follow-up", ("DATE", "Jan 5, 2024")),
        ("Discharged 5th March 2023", ("DATE", "5th March 2023")),
        ("A 93-year-old woman", ("AGE", "93")),
        ("92 yo male", ("AGE", "92")),
        ("Patient: John Smith\nDOB", ("PERSON", "John Smith")),
        ("Referred to Dr. Gonzalez for review", ("PERSON", "Gonzalez")),
    ],
)
def test_structured_rules(text, expected):
    assert expected in _found(text)


@pytest.mark.parametrize(
    "text",
    [
        "Card 4111 1111 1111 1112 declined",       # fails Luhn
        "The MRN is pending",                      # context word but no identifier
        "Account for 20% of the variance",         # 'account' as a verb
        "Upgrade to Python 3.10.12 today",         # version, not a dotted date
        "A 45-year-old man",                        # ages <= 89 are not identifiers
        "Black holes evaporate over 10^100 years",  # exponent, not an age
        "Released in 2023 after 13.8 billion years",
        "Version 999.1.1.1 shipped",               # not a valid IPv4
    ],
)
def test_structured_rules_reject_non_phi(text):
    assert detect_structured_phi(text) == []


def test_sample_corpora_have_almost_no_structured_false_positives():
    hits = []
    for path in sorted(SAMPLES.rglob("*.md")):
        hits += [(path.name, s.entity_type, s.text) for s in detect_structured_phi(path.read_text())]
    # The only identifier-shaped string in the non-PHI samples is "March 2023"
    # (a release date): month+year is a Safe Harbor date element.
    assert hits == [("transformers.md", "DATE", "March 2023")]


def test_overlaps_keep_longest_span():
    spans = [
        PhiSpan(0, 10, "PERSON", "John Smith"),
        PhiSpan(5, 10, "PERSON", "Smith"),
        PhiSpan(20, 31, "DATE", "Jan 5, 2024"),
        PhiSpan(20, 25, "DATE", "Jan 5"),
    ]
    assert resolve_overlaps(spans) == [spans[0], spans[2]]


def test_redact_structured_phi_for_logs():
    assert redact_structured_phi("lookup MRN: 99887766 ssn 123-45-6789") == "lookup MRN: [MRN] ssn [SSN]"


@pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_lg not installed")
class TestNer:
    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def _load(cls):
        phi.load_ner("en_core_web_lg")

    def test_names_and_places_detected(self):
        text = "John Smith was transferred from Springfield to see Maria Gonzalez."
        found = {(s.entity_type, s.text) for s in detect_phi(text)}
        assert {("PERSON", "John Smith"), ("PERSON", "Maria Gonzalez"), ("LOCATION", "Springfield")} <= found

    def test_names_propagate_within_document(self):
        text = (
            "Patient John Smith presented with chest pain.\n\n"
            "Plan discussed with the family. smith tolerated the procedure; JOHN SMITH discharged home."
        )
        person_texts = [s.text for s in detect_phi(text) if s.entity_type == "PERSON"]
        # "JOHN SMITH" is missed by NER and caught by propagation of "John Smith".
        assert {"John Smith", "JOHN SMITH"} <= set(person_texts)

    def test_organisations_and_generic_dates_are_kept(self):
        spans = detect_phi("Mercy Hospital reported results three days later in 2023.")
        assert all(s.entity_type not in {"DATE"} for s in spans)
        assert "Mercy Hospital" not in {s.text for s in spans}


def test_label_name_does_not_run_into_next_field():
    text = "Patient: Maria Gonzalez   MRN: 48213377"
    names = [sp.text for sp in phi.detect_structured_phi(text) if sp.entity_type == "PERSON"]
    assert names == ["Maria Gonzalez"]
