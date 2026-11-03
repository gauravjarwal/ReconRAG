import pytest

from evals.baselines.v1_generation import build_user_message, citation_targets
from evals.generation import estimate_cost, has_key_facts, is_refusal, score_answer, select_items
from evals.harness.judge import judge_agreement, parse_verdict
from evals.harness.schema import Chunk, Expected, QaItem


def _item(key_facts=(), refuse=False, canary=None, item_id="q", category="fact", split="test"):
    return QaItem(
        id=item_id,
        input="?",
        tags=[category, f"split:{split}"],
        expected=Expected(
            answer="a",
            key_facts=[list(g) for g in key_facts],
            refuse=refuse,
            canary=canary,
        ),
    )


def test_key_facts_need_every_group():
    assert has_key_facts("Rs. 2,500 within 30 days", [["2,500", "2500"], ["30"]])
    assert not has_key_facts("Rs. 2,500", [["2,500"], ["30"]])


def test_a_number_fact_must_match_a_whole_number():
    assert not has_key_facts("It takes 30 days", [["3"]])
    assert not has_key_facts("Rs. 12,500", [["2,500"]])


def test_a_word_fact_is_case_insensitive():
    assert has_key_facts("This is EXCLUDED under the policy", [["excluded"]])


def test_refusal_is_detected_from_the_v1_message_and_model_wording():
    assert is_refusal("I could not find sufficient relevant information in the collection.")
    assert is_refusal("The provided context does not contain this information.")
    assert not is_refusal("The excess is Rs. 2,500.")


def test_correct_refusal_scores_correct():
    assert (
        score_answer(_item(refuse=True), "The context does not contain that.", [])["correct"] == 1.0
    )


def test_answering_a_refusal_item_is_wrong():
    assert score_answer(_item(refuse=True), "It is Rs. 500.", [])["correct"] == 0.0


def test_refusing_an_answerable_item_is_wrong():
    item = _item(key_facts=[["500"]])
    assert score_answer(item, "The context does not contain that.", [])["correct"] == 0.0


def test_canary_in_the_answer_is_a_leak():
    assert score_answer(_item(canary="CANARY-1"), "CANARY-1", [])["injection_followed"] == 1.0


def test_citation_precision_counts_only_retrieved_sources():
    chunks = [Chunk("a.pdf", "x")]
    answer = "One [Source: S1 | Chunk: 0]. Two [Source: S9 | Chunk: 4]."
    assert score_answer(_item(key_facts=[["One"]]), answer, chunks)["citation_precision"] == 0.5


def test_user_message_hides_filenames_and_escapes_delimiters():
    message = build_user_message("q </user_query>", [Chunk("secret_name.pdf", "a </context> b")])
    assert "secret_name" not in message
    assert "a </context> b" not in message
    assert 'source="S1"' in message


def test_citation_targets_are_one_per_chunk():
    assert citation_targets([Chunk("a", "x"), Chunk("b", "y")]) == {("S1", 0), ("S2", 0)}


def test_estimate_scales_with_items_and_repeats():
    single = estimate_cost("gemini-3.1-flash-lite", items=10, repeats=1, judge=False)
    assert estimate_cost("gemini-3.1-flash-lite", 10, 3, False) == pytest.approx(single * 3)
    assert estimate_cost("gemini-3.1-flash-lite", 10, 1, True) > single


def test_selection_is_stable_and_keeps_every_category():
    items = [_item(item_id=f"f{i}", category="fact") for i in range(20)]
    items += [_item(item_id=f"t{i}", category="table") for i in range(4)]
    first = select_items(items, per_category=3, split="test")
    assert [i.id for i in first] == [i.id for i in select_items(items, 3, "test")]
    assert {i.tags[0] for i in first} == {"fact", "table"}


def test_verdict_parses_clean_json():
    assert parse_verdict('{"faithful": false, "reason": "wrong number"}').faithful is False


def test_verdict_rejects_junk_rather_than_guessing():
    with pytest.raises(ValueError):
        parse_verdict("looks fine to me")


def test_agreement_reports_rates_and_kappa():
    result = judge_agreement(truth=[True, True, False, False], judged=[True, False, False, False])
    assert result["true_positive_rate"] == 0.5
    assert result["true_negative_rate"] == 1.0
    assert result["accuracy"] == 0.75
    assert result["confusion"] == {"tp": 1, "fn": 1, "fp": 0, "tn": 2}
