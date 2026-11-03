from uuid import uuid4

from reconrag.config import Settings
from reconrag.domain import Hit
from reconrag.retrieval.context import _overlap_free, assemble
from reconrag.retrieval.fusion import suppress_identical, weighted_rrf
from reconrag.retrieval.gate import decide, term_coverage
from reconrag.retrieval.search import identifiers


def hit(name: str = "a", text: str = "text", **fields) -> Hit:
    return Hit(
        chunk_id=uuid4(),
        version_id=uuid4(),
        document_id=uuid4(),
        filename=name,
        ordinal=0,
        text=text,
        context="",
        section_path=[],
        page_start=None,
        page_end=None,
        kind="text",
        text_hash=text.encode(),
        **fields,
    )


def test_identifiers_finds_codes_not_words():
    assert identifiers("What does AC-200 cover under HIPAA in form cf7731 today?") == [
        "AC-200",
        "HIPAA",
        "cf7731",
    ]


def test_rrf_rewards_appearing_in_both_lists():
    both, dense_only, lexical_only = hit("both"), hit("dense"), hit("lex")
    fused = weighted_rrf([([dense_only, both], 1.0), ([lexical_only, both], 1.0)], 60)
    assert fused[0] is both


def test_rrf_weight_changes_the_winner():
    a, b = hit("a"), hit("b")
    fused = weighted_rrf([([a], 0.1), ([b], 1.0)], 60)
    assert fused[0] is b


def test_rrf_keeps_scores_from_each_list():
    first = hit("x", dense_similarity=0.8)
    second = hit("x", lexical_score=0.4)
    second.chunk_id = first.chunk_id
    fused = weighted_rrf([([first], 1.0), ([second], 1.0)], 60)
    assert fused[0].dense_similarity == 0.8
    assert fused[0].lexical_score == 0.4


def test_identical_text_keeps_the_best_and_names_the_others():
    best, copy, other = hit("a", "same"), hit("b", "same"), hit("c", "different")
    kept = suppress_identical([best, copy, other])
    assert kept == [best, other]
    assert best.signals["also_in"] == ["b"]


def gate_settings() -> Settings:
    return Settings(_env_file=None, gate_min_dense=0.5, gate_min_rerank=0.0)  # type: ignore[call-arg]


def test_gate_refuses_with_no_candidates():
    assert decide("q", [], gate_settings()).reason == "no_candidates"


def test_gate_uses_rerank_score_when_present():
    assert decide("q", [hit(rerank_score=-1.0)], gate_settings()).refuse
    assert not decide("q", [hit(rerank_score=1.0)], gate_settings()).refuse


def test_gate_works_without_a_reranker():
    weak = hit(text="nothing related", dense_similarity=0.2)
    assert decide("deductible amount", [weak], gate_settings()).reason == "low_similarity"
    strong = hit(dense_similarity=0.9)
    assert not decide("deductible amount", [strong], gate_settings()).refuse


def test_term_coverage_can_pass_a_low_similarity_hit():
    match = hit(text="The deductible amount is listed here", dense_similarity=0.1)
    assert term_coverage("what is the deductible amount?", match) == 1.0
    assert not decide("what is the deductible amount?", [match], gate_settings()).refuse


def test_budget_keeps_hits_before_neighbours():
    first, second = hit(text="one two three"), hit(text="four five six")
    around = {first.chunk_id: [{"ordinal": 1, "text": "extra " * 50}]}
    passages = assemble([first, second], around, max_tokens=10)
    assert [p.ref for p in passages] == ["S1", "S2"]
    assert "extra" not in passages[0].text


def test_neighbour_is_added_in_document_order():
    middle = hit(text="middle")
    middle.ordinal = 5
    around = {middle.chunk_id: [{"ordinal": 4, "text": "before"}, {"ordinal": 6, "text": "after"}]}
    assert assemble([middle], around, 100)[0].text == "before middle after"


def test_neighbour_already_chosen_is_not_repeated():
    a, b = hit(text="alpha"), hit(text="beta")
    b.version_id = a.version_id
    b.ordinal = 1
    around = {a.chunk_id: [{"ordinal": 1, "text": "beta"}]}
    assert assemble([a, b], around, 100)[0].text == "alpha"


def test_oversized_first_hit_is_still_used():
    assert len(assemble([hit(text="word " * 50)], {}, 5)) == 1


def test_overlap_is_removed_when_joining():
    shared = "this sentence is shared by both."
    assert _overlap_free(f"first part {shared}", f"{shared} last part") == (
        f"first part {shared} last part"
    )
