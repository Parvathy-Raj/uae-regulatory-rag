"""Run the golden set through one or more retrieval modes and score everything."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from uae_rag.config import RetrievalMode, Settings
from uae_rag.evaluation.golden import GoldenItem
from uae_rag.evaluation.judge import StructuredLLM, judge_correctness, judge_faithfulness
from uae_rag.evaluation.metrics import (
    ItemResult,
    Report,
    aggregate,
    answer_metrics,
    retrieval_metrics,
)
from uae_rag.generate.answer import generate_answer
from uae_rag.llm.client import LLMError
from uae_rag.pipeline import RAGPipeline
from uae_rag.schemas import ScoredChunk

Progress = Callable[[str], None]


class ModeRun(BaseModel):
    mode: RetrievalMode
    results: list[ItemResult]
    report: Report


def score_item(
    item: GoldenItem,
    chunks: list[ScoredChunk],
    settings: Settings,
    mode: RetrievalMode,
    gen_llm: StructuredLLM | None,
    judge_llm: StructuredLLM | None,
    retrieval_secs: float,
) -> ItemResult:
    retrieved = [s.chunk.chunk_id for s in chunks]
    metrics = retrieval_metrics(item, retrieved)
    res = ItemResult(
        item_id=item.id,
        category=item.category,
        question=item.question,
        expected_answer=item.expected_answer,
        answer="",
        refused=False,
        retrieved_ids=retrieved,
        cited_ids=[],
        supporting_ids=item.supporting_chunk_ids,
        context=[s.chunk.text for s in chunks],
        latency_ms={"retrieve": retrieval_secs * 1000},
        metrics=metrics,
    )
    if gen_llm is None:
        return res
    try:
        t0 = time.perf_counter()
        ans = generate_answer(item.question, chunks, gen_llm, settings.generation, mode)
        res.latency_ms["generate"] = (time.perf_counter() - t0) * 1000
    except LLMError as exc:
        res.error = f"generation: {exc}"
        return res
    res.answer, res.refused, res.refusal_reason = ans.answer, ans.refused, ans.refusal_reason
    res.cited_ids = ans.citations
    res.metrics |= answer_metrics(item, ans.refused, ans.citations)
    answerable = item.category != "unanswerable"
    res.metrics["faithfulness"] = None
    res.metrics["answer_correctness"] = 0.0 if (answerable and ans.refused) else None
    if judge_llm is None or ans.refused:
        return res
    try:
        cited_text = [
            s.chunk.text for s in chunks if s.chunk.chunk_id in set(ans.citations)
        ] or res.context
        faith = judge_faithfulness(judge_llm, ans.answer, cited_text)
        res.metrics["faithfulness"] = faith.score
        res.judge["faithfulness_claims"] = [c.model_dump() for c in faith.claims]
        if answerable:
            corr = judge_correctness(judge_llm, item.question, ans.answer, item.expected_answer)
            res.metrics["answer_correctness"] = corr.score
            res.judge["correctness"] = corr.model_dump()
    except LLMError as exc:
        res.error = f"judge: {exc}"
    return res


def run_mode(
    pipeline: RAGPipeline,
    items: Sequence[GoldenItem],
    mode: RetrievalMode,
    gen_llm: StructuredLLM | None,
    judge_llm: StructuredLLM | None,
    progress: Progress = lambda _msg: None,
) -> ModeRun:
    results: list[ItemResult] = []
    for i, item in enumerate(items, start=1):
        t0 = time.perf_counter()
        chunks = pipeline.retrieve(item.question, mode)
        results.append(
            score_item(
                item, chunks, pipeline.settings, mode, gen_llm, judge_llm, time.perf_counter() - t0
            )
        )
        progress(f"[{mode}] {i}/{len(items)}")
    return ModeRun(mode=mode, results=results, report=aggregate(results))


def write_mode_run(run: ModeRun, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"items_{run.mode}.jsonl").open("w", encoding="utf-8") as fh:
        for res in run.results:
            fh.write(res.model_dump_json() + "\n")
    (out_dir / f"report_{run.mode}.json").write_text(
        run.model_dump_json(include={"mode", "report"}, indent=2), encoding="utf-8"
    )
