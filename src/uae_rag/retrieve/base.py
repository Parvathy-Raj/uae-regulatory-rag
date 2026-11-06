"""Retriever interface and reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from uae_rag.schemas import ScoredChunk


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60, names: Sequence[str] | None = None
) -> list[ScoredChunk]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank), rank starting at 1."""
    names = names or [f"list{i}" for i in range(len(rankings))]
    fused: dict[str, ScoredChunk] = {}
    for name, ranking in zip(names, rankings, strict=True):
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.chunk_id
            if cid not in fused:
                fused[cid] = ScoredChunk(chunk=item.chunk, score=0.0)
            entry = fused[cid]
            entry.score += 1.0 / (k + rank)
            entry.stage_scores[name] = item.score
            entry.stage_scores[f"{name}_rank"] = float(rank)
    return sorted(fused.values(), key=lambda s: s.score, reverse=True)
