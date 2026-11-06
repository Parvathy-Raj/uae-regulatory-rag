"""Grounded answer generation with citations and refusal.

The model sees numbered context blocks [1]..[n] and must cite them inline. Short
numeric labels are far more reliable than asking a model to copy long chunk ids;
we map the labels back to chunk_ids afterwards and drop any label that does not
exist (an invented citation).
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field

from uae_rag.config import GenerationConfig, RetrievalMode
from uae_rag.schemas import ScoredChunk

REFUSAL_TEXT = (
    "I can't answer that from the indexed documents. The retrieved passages don't contain "
    "enough information, so I'd rather not guess."
)

SYSTEM_PROMPT = """You answer questions about UAE tax using ONLY the numbered context passages.
Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Cite every factual sentence with the passage number in square brackets, e.g. [2] or [1][3].
- If the passages do not contain the answer, set "refused" to true and leave "answer" empty.
- Be concise: 1-5 sentences. Quote figures (rates, thresholds, dates) exactly as written.
- Output fields: "answer" (string), "citations" (list of passage numbers you used),
  "refused" (boolean), "refusal_reason" (string, empty unless refused)."""


class AnswerDraft(BaseModel):
    """What the LLM returns."""

    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


class Answer(BaseModel):
    """Final answer returned to callers (CLI, eval, UI)."""

    question: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    citations: list[str] = Field(default_factory=list)  # chunk_ids
    invalid_citations: list[int] = Field(default_factory=list)
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    mode: RetrievalMode | None = None
    model: str = ""
    latency_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def retrieved_ids(self) -> list[str]:
        return [s.chunk.chunk_id for s in self.retrieved]


class StructuredLLM(Protocol):
    model: str

    def structured(
        self, messages: list[dict[str, str]], schema: type[AnswerDraft]
    ) -> AnswerDraft: ...


def format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for i, item in enumerate(chunks, start=1):
        c = item.chunk
        pages = (
            f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        )
        blocks.append(f"[{i}] ({c.source_title}, {pages}, section: {c.heading})\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[ScoredChunk]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}",
        },
    ]


def labels_in_text(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d{1,2})\]", text)]


def map_citations(draft: AnswerDraft, chunks: list[ScoredChunk]) -> tuple[list[str], list[int]]:
    """Return (valid chunk_ids in first-cited order, invalid labels)."""
    labels = list(dict.fromkeys([*labels_in_text(draft.answer), *draft.citations]))
    valid, invalid = [], []
    for label in labels:
        if 1 <= label <= len(chunks):
            valid.append(chunks[label - 1].chunk.chunk_id)
        else:
            invalid.append(label)
    return list(dict.fromkeys(valid)), invalid


def below_threshold(chunks: list[ScoredChunk], mode: RetrievalMode, cfg: GenerationConfig) -> bool:
    """Pre-generation refusal: nothing retrieved, or the best score is too weak to trust."""
    if not chunks:
        return True
    best = chunks[0].score
    if mode == "hybrid_rerank":
        return best < cfg.min_rerank_score
    if mode == "dense":
        return best < cfg.min_dense_score
    return False  # RRF scores are rank-based, not calibrated; no threshold.


def refusal(
    question: str, reason: str, chunks: list[ScoredChunk], mode: RetrievalMode | None, model: str
) -> Answer:
    return Answer(
        question=question,
        answer=REFUSAL_TEXT,
        refused=True,
        refusal_reason=reason,
        retrieved=chunks,
        mode=mode,
        model=model,
    )


def generate_answer(
    question: str,
    chunks: list[ScoredChunk],
    llm: StructuredLLM,
    cfg: GenerationConfig,
    mode: RetrievalMode | None = None,
) -> Answer:
    if mode is not None and below_threshold(chunks, mode, cfg):
        return refusal(question, "low_retrieval_score", chunks, mode, llm.model)
    draft = llm.structured(build_messages(question, chunks), AnswerDraft)
    if draft.refused or not draft.answer.strip():
        return refusal(question, draft.refusal_reason or "model_refused", chunks, mode, llm.model)
    valid, invalid = map_citations(draft, chunks)
    if cfg.refuse_if_no_valid_citation and not valid:
        return refusal(question, "no_valid_citation", chunks, mode, llm.model)
    return Answer(
        question=question,
        answer=draft.answer.strip(),
        refused=False,
        citations=valid,
        invalid_citations=invalid,
        retrieved=chunks,
        mode=mode,
        model=llm.model,
    )
