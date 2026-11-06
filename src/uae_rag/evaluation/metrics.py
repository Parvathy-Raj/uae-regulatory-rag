"""Pure metric functions plus per-category aggregation.

Every metric returns None when it does not apply to an item (e.g. recall for an
unanswerable question). None values are skipped when averaging, and the number of
items behind each average is reported next to it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, Field

from uae_rag.evaluation.golden import GoldenItem

METRICS = (
    "recall_at_5",
    "mrr",
    "faithfulness",
    "answer_correctness",
    "refusal_accuracy",
    "false_refusal",
    "citation_accuracy",
)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float | None:
    if not relevant:
        return None
    top = set(retrieved[:k])
    return sum(1 for r in relevant if r in top) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float | None:
    if not relevant:
        return None
    wanted = set(relevant)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in wanted:
            return 1.0 / rank
    return 0.0


def citation_hit(cited: Sequence[str], relevant: Sequence[str], refused: bool) -> float | None:
    """1 if at least one citation points at a gold supporting chunk. Only for answered,
    answerable questions."""
    if refused or not relevant:
        return None
    return 1.0 if set(cited) & set(relevant) else 0.0


def refusal_correct(refused: bool, category: str) -> float | None:
    """Unanswerable items: did the system refuse? (1 = correct)."""
    return (1.0 if refused else 0.0) if category == "unanswerable" else None


def false_refusal(refused: bool, category: str) -> float | None:
    """Answerable items: did the system wrongly refuse? (lower is better)."""
    return None if category == "unanswerable" else (1.0 if refused else 0.0)


class ItemResult(BaseModel):
    item_id: str
    category: str
    question: str
    expected_answer: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    retrieved_ids: list[str]
    cited_ids: list[str]
    supporting_ids: list[str]
    context: list[str] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    judge: dict[str, object] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    error: str = ""


def retrieval_metrics(item: GoldenItem, retrieved_ids: Sequence[str]) -> dict[str, float | None]:
    return {
        "recall_at_5": recall_at_k(retrieved_ids, item.supporting_chunk_ids, 5),
        "mrr": reciprocal_rank(retrieved_ids, item.supporting_chunk_ids),
    }


def answer_metrics(
    item: GoldenItem, refused: bool, cited: Sequence[str]
) -> dict[str, float | None]:
    return {
        "refusal_accuracy": refusal_correct(refused, item.category),
        "false_refusal": false_refusal(refused, item.category),
        "citation_accuracy": citation_hit(cited, item.supporting_chunk_ids, refused),
    }


class MetricSummary(BaseModel):
    mean: float | None
    n: int


Report = dict[str, dict[str, MetricSummary]]  # slice -> metric -> summary


def _summarise(values: list[float]) -> MetricSummary:
    return MetricSummary(mean=sum(values) / len(values) if values else None, n=len(values))


def aggregate(results: Sequence[ItemResult]) -> Report:
    """Return {"all": {...}, "easy": {...}, ...} with mean and count per metric."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for res in results:
        for name, value in res.metrics.items():
            if value is None:
                continue
            buckets["all"][name].append(value)
            buckets[res.category][name].append(value)
    report: Report = {}
    for slice_name in ["all", *sorted(k for k in buckets if k != "all")]:
        report[slice_name] = {m: _summarise(buckets[slice_name].get(m, [])) for m in METRICS}
    return report


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Agreement between two raters beyond chance. 1 = perfect, 0 = chance level."""
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
