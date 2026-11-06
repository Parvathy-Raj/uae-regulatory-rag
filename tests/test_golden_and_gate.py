import random

from conftest import make_chunk
from uae_rag.evaluation.calibration import HumanLabel, compare
from uae_rag.evaluation.gate import GateConfig, evaluate_gate, gate_markdown
from uae_rag.evaluation.golden import (
    UNANSWERABLE_ANSWER,
    DraftQA,
    GoldenItem,
    apply_decision,
    draft_items,
    merge_drafts,
    pick_pair,
    plan_counts,
    read_golden,
    stratified_subset,
    write_golden,
)
from uae_rag.evaluation.metrics import ItemResult

LONG = " ".join(["word"] * 80)


def item(i: int, cat: str = "easy", status: str = "approved") -> GoldenItem:
    return GoldenItem(
        id=f"id{i}", question=f"q{i}", expected_answer="a", category=cat, status=status
    )


def test_plan_counts_sum_to_n():
    counts = plan_counts(100)
    assert sum(counts.values()) == 100 and counts["easy"] == 60


def test_pick_pair_requires_same_doc_different_section():
    a = make_chunk("d:000:00", LONG, heading="A")
    b = make_chunk("d:001:00", LONG, heading="B")
    other_doc = make_chunk("x:000:00", LONG, heading="C", source="x")
    pair = pick_pair([a], random.Random(0), neighbours=lambda c: [a, other_doc, b])
    assert pair == (a, b)


def test_draft_items_shapes_each_category():
    chunks = [make_chunk(f"d:{i:03d}:00", LONG, heading=f"H{i}") for i in range(6)]
    items = draft_items(
        chunks,
        20,
        draft=lambda cat, cs: DraftQA(question=f"{cat}?", expected_answer="ans"),
        neighbours=lambda c: chunks,
    )
    by_cat = {i.category for i in items}
    assert by_cat == {"easy", "multi_section", "unanswerable"}
    for it in items:
        assert it.status == "pending"
        if it.category == "unanswerable":
            assert it.supporting_chunk_ids == [] and it.expected_answer == UNANSWERABLE_ANSWER
        if it.category == "multi_section":
            assert len(it.supporting_chunk_ids) == 2


def test_apply_decision_and_merge_keep_human_work(tmp_path):
    pending = item(1, status="pending")
    edited = apply_decision(pending, "edit", {"question": "better?", "status": "rejected"})
    assert edited.status == "approved" and edited.question == "better?"
    assert apply_decision(pending, "reject").status == "rejected"
    merged = merge_drafts([edited], [item(1, status="pending"), item(2, status="pending")])
    assert [m.id for m in merged] == ["id1", "id2"] and merged[0].question == "better?"
    path = tmp_path / "g.jsonl"
    write_golden(merged, path)
    assert read_golden(path) == merged


def test_stratified_subset_keeps_every_category():
    items = (
        [item(i, "easy") for i in range(60)]
        + [item(100 + i, "multi_section") for i in range(25)]
        + [item(200 + i, "unanswerable") for i in range(15)]
    )
    sub = stratified_subset(items, 15)
    cats = [s.category for s in sub]
    assert set(cats) == {"easy", "multi_section", "unanswerable"}
    assert 13 <= len(sub) <= 17


def plain(mean: float, n: int = 10) -> dict:
    return {"mean": mean, "n": n}


def test_gate_passes_within_threshold_and_fails_beyond():
    cfg = GateConfig(max_drop={"recall_at_5": 0.05}, min_items=3)
    base = {"all": {"recall_at_5": plain(0.80)}}
    assert evaluate_gate({"all": {"recall_at_5": plain(0.76)}}, base, cfg).passed
    failed = evaluate_gate({"all": {"recall_at_5": plain(0.70)}}, base, cfg)
    assert not failed.passed and "dropped" in failed.checks[0].note
    assert "❌" in gate_markdown(failed, "table")


def test_gate_ignores_tiny_slices_and_missing_baseline():
    cfg = GateConfig(max_drop={"faithfulness": 0.05}, min_items=3)
    base = {"easy": {"faithfulness": plain(0.9)}}
    assert evaluate_gate({"easy": {"faithfulness": plain(0.1, n=2)}}, base, cfg).passed
    assert evaluate_gate({"all": {}}, None, cfg).message.startswith("No baseline")


def test_calibration_compares_human_and_judge():
    results = [
        ItemResult(
            item_id="a",
            category="easy",
            question="q",
            expected_answer="e",
            answer="x",
            refused=False,
            retrieved_ids=[],
            cited_ids=[],
            supporting_ids=[],
            metrics={"faithfulness": 1.0},
            judge={"correctness": {"verdict": "correct"}},
        ),
        ItemResult(
            item_id="b",
            category="easy",
            question="q",
            expected_answer="e",
            answer="x",
            refused=False,
            retrieved_ids=[],
            cited_ids=[],
            supporting_ids=[],
            metrics={"faithfulness": 0.5},
            judge={"correctness": {"verdict": "partial"}},
        ),
    ]
    labels = [
        HumanLabel(item_id="a", mode="m", faithfulness="faithful", correctness="correct"),
        HumanLabel(item_id="b", mode="m", faithfulness="faithful", correctness="partial"),
    ]
    faith, corr = compare(labels, results)
    assert faith.n == 2 and faith.agreement == 0.5
    assert faith.confusion == {"faithful->faithful": 1, "faithful->unfaithful": 1}
    assert corr.agreement == 1.0
