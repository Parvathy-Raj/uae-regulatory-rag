import pytest

from uae_rag.evaluation.metrics import (
    ItemResult,
    aggregate,
    citation_hit,
    cohen_kappa,
    false_refusal,
    recall_at_k,
    reciprocal_rank,
    refusal_correct,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b"], k=5) == 1.0
    assert recall_at_k(["a", "b", "c", "d", "e", "f"], ["f", "a"], k=5) == 0.5
    assert recall_at_k(["a"], [], k=5) is None


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["a"]) == 0.0
    assert reciprocal_rank(["x"], []) is None


def test_refusal_metrics_apply_to_the_right_category():
    assert refusal_correct(True, "unanswerable") == 1.0
    assert refusal_correct(False, "unanswerable") == 0.0
    assert refusal_correct(True, "easy") is None
    assert false_refusal(True, "easy") == 1.0
    assert false_refusal(True, "unanswerable") is None


def test_citation_hit():
    assert citation_hit(["a", "z"], ["a"], refused=False) == 1.0
    assert citation_hit(["z"], ["a"], refused=False) == 0.0
    assert citation_hit([], ["a"], refused=True) is None


def res(cat: str, **metrics) -> ItemResult:
    return ItemResult(
        item_id="i",
        category=cat,
        question="q",
        expected_answer="e",
        answer="a",
        refused=False,
        retrieved_ids=[],
        cited_ids=[],
        supporting_ids=[],
        metrics=metrics,
    )


def test_aggregate_slices_by_category_and_skips_none():
    report = aggregate(
        [
            res("easy", recall_at_5=1.0),
            res("easy", recall_at_5=0.0),
            res("unanswerable", recall_at_5=None, refusal_accuracy=1.0),
        ]
    )
    assert report["all"]["recall_at_5"].mean == 0.5 and report["all"]["recall_at_5"].n == 2
    assert report["easy"]["recall_at_5"].n == 2
    assert report["unanswerable"]["recall_at_5"].mean is None
    assert report["unanswerable"]["refusal_accuracy"].mean == 1.0
    assert next(iter(report)) == "all"


def test_cohen_kappa_known_values():
    assert cohen_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == 1.0
    # observed 0.5, expected 0.5 -> kappa 0
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == pytest.approx(0.0)
    assert cohen_kappa([], []) is None
