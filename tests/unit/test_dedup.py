import pytest

from reconrag.ingestion.dedup import (
    document_key,
    edition,
    hamming,
    simhash,
    text_digest,
    undated_key,
)


def test_text_digest_ignores_case_and_spacing():
    assert text_digest("Hello   World") == text_digest("hello world")


def test_simhash_fits_signed_bigint():
    value = simhash("some text about insurance plans and their limits")
    assert -(2**63) <= value < 2**63


def test_near_text_is_close_and_different_text_is_far():
    base = " ".join(f"clause {i} covers event type {i * 3} under normal terms" for i in range(60))
    edited = base.replace("clause 30 covers", "clause 30 excludes")
    other = " ".join(f"invoice {i} was paid by vendor {i * 7} in full" for i in range(60))
    assert hamming(simhash(base), simhash(edited)) <= 6
    assert hamming(simhash(base), simhash(other)) > 15


@pytest.mark.parametrize(
    "name",
    [
        "Policy.pdf",
        "policy_v2.pdf",
        "Policy-final.pdf",
        "policy (1).pdf",
        "policy v3.1 FINAL.docx",
        "Policy_rev2_draft.pdf",
    ],
)
def test_version_noise_maps_to_one_key(name: str):
    assert document_key(name) == "policy"


def test_distinct_documents_keep_distinct_keys():
    assert document_key("gold_plan.pdf") != document_key("silver_plan.pdf")
    assert document_key("annual report 2024.pdf") != document_key("annual report 2025.pdf")


def test_name_that_is_only_version_words_still_has_a_key():
    assert document_key("final.pdf") == "final"


def test_edition_reads_dates_from_names():
    assert edition("rates_2026.pdf") == 20260101
    assert edition("rates_2026-03.pdf") == 20260301
    assert edition("rates 2026-03-15.pdf") == 20260315
    assert edition("rates.pdf") == 0


def test_undated_key_groups_a_series():
    assert undated_key("rates 2025") == undated_key("rates 2026") == "rates"
