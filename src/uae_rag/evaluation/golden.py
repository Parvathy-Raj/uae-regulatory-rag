"""Golden set: LLM-drafted questions that a human approves, edits or rejects.

Drafts are written to eval/golden.jsonl with status "pending". Only items marked
"approved" by a human are ever used for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from uae_rag.schemas import Chunk

Category = Literal["easy", "multi_section", "unanswerable"]
CATEGORIES: tuple[Category, ...] = ("easy", "multi_section", "unanswerable")
Status = Literal["pending", "approved", "rejected"]
DEFAULT_MIX: dict[Category, float] = {"easy": 0.6, "multi_section": 0.25, "unanswerable": 0.15}
UNANSWERABLE_ANSWER = "The documents do not contain this information."


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    category: Category
    status: Status = "pending"
    notes: str = ""
    drafted_by: str = ""


class DraftQA(BaseModel):
    """LLM output when drafting a question."""

    question: str
    expected_answer: str


DRAFT_PROMPTS: dict[Category, str] = {
    "easy": (
        "You write evaluation questions for a UAE tax Q&A system. Read the passage and write ONE "
        "specific question a taxpayer might ask that the passage answers fully. Then give the "
        "correct answer using only the passage, in 1-3 sentences. Avoid yes/no questions and "
        "avoid referring to 'the passage'."
    ),
    "multi_section": (
        "You write evaluation questions for a UAE tax Q&A system. Read the TWO passages, which come "
        "from different sections. Write ONE question whose full answer needs facts from BOTH "
        "passages, then answer it using only the passages, in 2-4 sentences. Do not mention "
        "'passage 1' or 'passage 2' in the question."
    ),
    "unanswerable": (
        "You write evaluation questions for a UAE tax Q&A system. Read the passage, then write ONE "
        "realistic, closely related question that the passage does NOT answer (for example a "
        "detail, case or figure it does not state). For expected_answer write exactly: "
        f"'{UNANSWERABLE_ANSWER}'"
    ),
}


def item_id(category: str, chunk_ids: Sequence[str], question: str) -> str:
    digest = hashlib.sha1(f"{category}|{'|'.join(chunk_ids)}|{question}".encode()).hexdigest()
    return f"{category[:4]}-{digest[:10]}"


def plan_counts(n: int, mix: dict[Category, float] = DEFAULT_MIX) -> dict[Category, int]:
    counts = {cat: int(n * share) for cat, share in mix.items()}
    counts["easy"] += n - sum(counts.values())
    return counts


def pick_pair(
    chunks: Sequence[Chunk], rng: random.Random, neighbours: Callable[[Chunk], list[Chunk]]
) -> tuple[Chunk, Chunk] | None:
    """A chunk plus a related chunk from a different section of the same document."""
    first = rng.choice(chunks)
    for other in neighbours(first):
        same_doc = other.source_id == first.source_id
        if same_doc and other.heading != first.heading and other.chunk_id != first.chunk_id:
            return first, other
    return None


def draft_items(
    chunks: Sequence[Chunk],
    n: int,
    draft: Callable[[Category, list[Chunk]], DraftQA],
    neighbours: Callable[[Chunk], list[Chunk]],
    seed: int = 7,
    min_words: int = 60,
    model_name: str = "",
) -> list[GoldenItem]:
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text.split()) >= min_words]
    items: list[GoldenItem] = []
    for category, count in plan_counts(n).items():
        made, tries = 0, 0
        while made < count and tries < count * 4:
            tries += 1
            if category == "multi_section":
                pair = pick_pair(pool, rng, neighbours)
                if pair is None:
                    continue
                source = list(pair)
            else:
                source = [rng.choice(pool)]
            qa = draft(category, source)
            supporting = [] if category == "unanswerable" else [c.chunk_id for c in source]
            expected = UNANSWERABLE_ANSWER if category == "unanswerable" else qa.expected_answer
            # Unanswerable items keep the seed chunk in notes so the reviewer can check it.
            notes = f"seed_chunk={source[0].chunk_id}" if category == "unanswerable" else ""
            items.append(
                GoldenItem(
                    id=item_id(category, [c.chunk_id for c in source], qa.question),
                    question=qa.question.strip(),
                    expected_answer=expected.strip(),
                    supporting_chunk_ids=supporting,
                    category=category,
                    notes=notes,
                    drafted_by=model_name,
                )
            )
            made += 1
    return items


def read_golden(path: Path) -> list[GoldenItem]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [GoldenItem.model_validate_json(line) for line in fh if line.strip()]


def write_golden(items: Sequence[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")


def merge_drafts(existing: Sequence[GoldenItem], new: Sequence[GoldenItem]) -> list[GoldenItem]:
    """Append new drafts without touching items a human already reviewed."""
    seen = {i.id for i in existing}
    return [*existing, *[i for i in new if i.id not in seen]]


def apply_decision(
    item: GoldenItem, decision: Literal["approve", "reject", "edit"], edits: dict | None = None
) -> GoldenItem:
    if decision == "reject":
        return item.model_copy(update={"status": "rejected"})
    update: dict = {"status": "approved"}
    if decision == "edit" and edits:
        allowed = {"question", "expected_answer", "supporting_chunk_ids", "category", "notes"}
        update |= {k: v for k, v in edits.items() if k in allowed}
    return GoldenItem.model_validate({**item.model_dump(), **update})


def approved(items: Sequence[GoldenItem]) -> list[GoldenItem]:
    return [i for i in items if i.status == "approved"]


def stratified_subset(items: Sequence[GoldenItem], n: int, seed: int = 13) -> list[GoldenItem]:
    """Pick about n items keeping the category mix (at least one per category present)."""
    by_cat: dict[str, list[GoldenItem]] = defaultdict(list)
    for item in items:
        by_cat[item.category].append(item)
    rng = random.Random(seed)
    total = len(items)
    chosen: list[GoldenItem] = []
    for _cat, group in sorted(by_cat.items()):
        take = max(1, round(n * len(group) / total)) if total else 0
        chosen.extend(rng.sample(group, min(take, len(group))))
    return chosen


def dataset_hash(items: Sequence[GoldenItem]) -> str:
    payload = json.dumps([i.model_dump() for i in items], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
