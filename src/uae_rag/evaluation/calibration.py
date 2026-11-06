"""Judge calibration: compare LLM-judge labels with human labels on the same answers."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from uae_rag.evaluation.judge import faithfulness_label
from uae_rag.evaluation.metrics import ItemResult, cohen_kappa


class HumanLabel(BaseModel):
    item_id: str
    mode: str
    faithfulness: Literal["faithful", "unfaithful"]
    correctness: Literal["correct", "partial", "incorrect"] | None = None


class Agreement(BaseModel):
    metric: str
    n: int
    agreement: float | None
    kappa: float | None
    confusion: dict[str, int]  # "human->judge" -> count


def read_results(path: Path) -> list[ItemResult]:
    with path.open(encoding="utf-8") as fh:
        return [ItemResult.model_validate_json(line) for line in fh if line.strip()]


def sample_for_labeling(
    results: Sequence[ItemResult], n: int = 30, seed: int = 21
) -> list[ItemResult]:
    """Answered items that the judge scored (refusals have nothing to judge)."""
    pool = [r for r in results if not r.refused and r.metrics.get("faithfulness") is not None]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def judge_labels(result: ItemResult) -> tuple[str, str | None]:
    faith = faithfulness_label(result.metrics.get("faithfulness"))
    corr = result.judge.get("correctness")
    verdict = corr.get("verdict") if isinstance(corr, dict) else None
    return faith, verdict


def _agreement(metric: str, pairs: list[tuple[str, str]]) -> Agreement:
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    agree = sum(h == j for h, j in pairs) / len(pairs) if pairs else None
    confusion = Counter(f"{h}->{j}" for h, j in pairs)
    return Agreement(
        metric=metric,
        n=len(pairs),
        agreement=agree,
        kappa=cohen_kappa(human, judge),
        confusion=dict(sorted(confusion.items())),
    )


def compare(labels: Sequence[HumanLabel], results: Sequence[ItemResult]) -> list[Agreement]:
    by_id = {r.item_id: r for r in results}
    faith_pairs: list[tuple[str, str]] = []
    corr_pairs: list[tuple[str, str]] = []
    for label in labels:
        res = by_id.get(label.item_id)
        if res is None:
            continue
        j_faith, j_corr = judge_labels(res)
        faith_pairs.append((label.faithfulness, j_faith))
        if label.correctness and j_corr:
            corr_pairs.append((label.correctness, j_corr))
    return [_agreement("faithfulness", faith_pairs), _agreement("answer_correctness", corr_pairs)]
