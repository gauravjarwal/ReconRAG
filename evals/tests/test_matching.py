from evals.harness.matching import contains_evidence, evidence_matches, evidence_rank, normalize
from evals.harness.schema import Evidence


def test_normalize_collapses_case_punctuation_and_whitespace():
    assert normalize("  The EXCESS,\n is:  Rs. 2,500 ") == "the excess is rs 2 500"


def test_evidence_found_despite_line_wrapped_hyphenation():
    chunk = "The policy holder must pay the deduct-\nible of Rs. 2,500 for each claim."
    assert contains_evidence(chunk, "pay the deductible of Rs. 2,500 for each claim")


def test_evidence_rejected_when_a_number_differs():
    chunk = "The policy holder must pay the deductible of Rs. 3,500 for each claim."
    assert not contains_evidence(chunk, "pay the deductible of Rs. 2,500 for each claim")


def test_evidence_rejected_for_an_unrelated_chunk():
    assert not contains_evidence("Cancellation needs written notice.", "theft of the device")


def test_evidence_rank_needs_the_right_file():
    ranked = [("other.pdf", "Excess is Rs. 2,500"), ("plan.pdf", "Excess is Rs. 2,500")]
    assert evidence_rank(ranked, Evidence(files=["plan.pdf"], quote="Excess is Rs. 2,500")) == 2


def test_evidence_rank_is_none_when_absent():
    evidence = Evidence(files=["plan.pdf"], quote="Excess is Rs. 2,500")
    assert evidence_rank([("plan.pdf", "nothing here")], evidence) is None


def test_table_row_evidence_found_when_cells_are_split_across_lines():
    chunk = "Screen repair\n3450\n5200"
    assert evidence_matches(chunk, Evidence(files=["a.xlsx"], row="Screen repair", value="3450"))


def test_table_row_evidence_ignores_thousands_separators():
    chunk = "Annual premium: Rs. 6,980"
    assert evidence_matches(chunk, Evidence(files=["a.pdf"], row="Annual premium", value="6980"))


def test_table_row_evidence_needs_the_row_label():
    chunk = "Accessories\n3450"
    assert not evidence_matches(
        chunk, Evidence(files=["a.xlsx"], row="Screen repair", value="3450")
    )


def test_quote_evidence_uses_the_quote():
    chunk = "Excess is Rs. 2,500."
    assert evidence_matches(chunk, Evidence(files=["a.pdf"], quote="Excess is Rs. 2,500"))


def test_a_number_elsewhere_in_the_chunk_does_not_rescue_a_wrong_value():
    chunk = "Notify us within 27 days of the event. Refunds take 30 days."
    assert not contains_evidence(chunk, "Notify us within 30 days of the event.")


def test_table_value_must_be_a_whole_number():
    chunk = "Document review\n13"
    assert not evidence_matches(chunk, Evidence(files=["a.pdf"], row="Document review", value="3"))


def test_a_fragment_shorter_than_the_quote_is_not_evidence():
    assert not contains_evidence("the excess is", "the excess is Rs. 500 for each claim")
