import math

import pytest

from evals.harness.metrics import (
    all_found_at_k,
    bootstrap_mean_ci,
    cohen_kappa,
    mrr,
    ndcg_at_k,
    paired_bootstrap,
    percentile,
    recall_at_k,
    refusal_scores,
)


def test_recall_counts_evidence_groups_found_within_k():
    assert recall_at_k([1, 7], k=5) == 0.5


def test_recall_treats_a_missing_group_as_not_found():
    assert recall_at_k([None, 2], k=5) == 0.5


def test_recall_rank_equal_to_k_counts():
    assert recall_at_k([5], k=5) == 1.0


def test_all_found_needs_every_group_within_k():
    assert all_found_at_k([1, 3], k=5) == 1.0
    assert all_found_at_k([1, 9], k=5) == 0.0


def test_mrr_uses_the_best_ranked_group():
    assert mrr([4, 2]) == 0.5


def test_mrr_is_zero_when_nothing_is_found():
    assert mrr([None, None]) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1, 2], k=5) == pytest.approx(1.0)


def test_ndcg_discounts_a_lower_rank():
    # One group at rank 3: DCG = 1/log2(4) = 0.5, ideal = 1/log2(2) = 1.
    assert ndcg_at_k([3], k=5) == pytest.approx(0.5)


def test_ndcg_counts_a_missing_group_against_the_ideal():
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k([1, None], k=5) == pytest.approx(expected)


def test_ndcg_stays_at_one_when_one_chunk_holds_both_groups():
    assert ndcg_at_k([1, 1], k=5) == pytest.approx(1.0)


def test_ndcg_ignores_ranks_beyond_k():
    assert ndcg_at_k([6], k=5) == 0.0


def test_refusal_scores_from_a_small_confusion_matrix():
    scores = refusal_scores(
        should_refuse=[True, True, False, False], did_refuse=[True, False, True, False]
    )
    assert (scores["tp"], scores["fn"], scores["fp"], scores["tn"]) == (1, 1, 1, 1)
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5


def test_refusal_precision_is_zero_when_nothing_is_refused():
    scores = refusal_scores(should_refuse=[True, False], did_refuse=[False, False])
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 50) == 5.0


def test_percentile_of_nothing_is_none():
    assert percentile([], 95) is None


def test_bootstrap_ci_contains_the_mean_and_is_repeatable():
    values = [0.0, 1.0] * 50
    first = bootstrap_mean_ci(values, seed=7)
    assert first == bootstrap_mean_ci(values, seed=7)
    assert first["low"] < first["mean"] < first["high"]


def test_paired_bootstrap_detects_a_consistent_gain():
    result = paired_bootstrap(a=[0.0] * 80 + [1.0] * 20, b=[1.0] * 80 + [1.0] * 20, seed=1)
    assert result["delta"] == pytest.approx(0.8)
    assert result["low"] > 0
    assert result["significant"] is True


def test_paired_bootstrap_calls_identical_runs_not_significant():
    values = [0.0, 1.0] * 50
    result = paired_bootstrap(values, values, seed=1)
    assert result["delta"] == 0.0
    assert result["significant"] is False


def test_paired_bootstrap_rejects_unpaired_lists():
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap([1.0], [1.0, 0.0])


def test_kappa_is_one_for_perfect_agreement():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_kappa_is_zero_when_agreement_matches_chance():
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_kappa_from_a_worked_example():
    # 3 of 4 agree (0.75). Chance: 0.75*0.5 + 0.25*0.5 = 0.5. Kappa = 0.25 / 0.5.
    assert cohen_kappa([True, True, True, False], [True, True, False, False]) == 0.5
