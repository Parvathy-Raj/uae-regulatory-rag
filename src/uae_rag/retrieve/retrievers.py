"""The three retrieval modes compared in Stage 1.

(a) DenseRetriever         - bge-small embeddings in Chroma
(b) HybridRetriever        - BM25 + dense fused with reciprocal rank fusion
(c) RerankRetriever        - hybrid top-N rescored by a cross-encoder
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

from uae_rag.retrieve.base import Retriever, reciprocal_rank_fusion
from uae_rag.schemas import Chunk, ScoredChunk

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "do",
        "does",
        "i",
        "my",
        "we",
        "our",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.index = BM25Okapi([tokenize(c.embed_text()) or ["_"] for c in self.chunks])

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self.index.get_scores(tokenize(query) or ["_"])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in order]


class DenseRetriever:
    """Searches a Chroma collection. Scores are cosine similarity (1 - distance)."""

    name = "dense"

    def __init__(
        self, collection, embed_query: Callable[[str], list[float]], by_id: dict[str, Chunk]
    ):
        self.collection = collection
        self.embed_query = embed_query
        self.by_id = by_id

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        res = self.collection.query(
            query_embeddings=[self.embed_query(query)], n_results=k, include=["distances"]
        )
        ids, dists = res["ids"][0], res["distances"][0]
        return [
            ScoredChunk(
                chunk=self.by_id[cid], score=1.0 - float(d), stage_scores={"dense": 1.0 - float(d)}
            )
            for cid, d in zip(ids, dists, strict=True)
            if cid in self.by_id
        ]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: Retriever, bm25: Retriever, candidate_k: int = 30, rrf_k: int = 60):
        self.dense, self.bm25 = dense, bm25
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        fused = reciprocal_rank_fusion(
            [
                self.dense.retrieve(query, self.candidate_k),
                self.bm25.retrieve(query, self.candidate_k),
            ],
            k=self.rrf_k,
            names=["dense", "bm25"],
        )
        return fused[:k]


class RerankRetriever:
    """Rescore the first-stage pool with a cross-encoder; score is in [0, 1]."""

    name = "hybrid_rerank"

    def __init__(
        self,
        first_stage: Retriever,
        score_pairs: Callable[[list[tuple[str, str]]], list[float]],
        candidate_k: int = 30,
    ):
        self.first_stage = first_stage
        self.score_pairs = score_pairs
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        pool = self.first_stage.retrieve(query, self.candidate_k)
        if not pool:
            return []
        scores = self.score_pairs([(query, s.chunk.embed_text()) for s in pool])
        rescored = []
        for item, score in zip(pool, scores, strict=True):
            stages = {**item.stage_scores, "first_stage": item.score, "rerank": float(score)}
            rescored.append(ScoredChunk(chunk=item.chunk, score=float(score), stage_scores=stages))
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:k]


def cross_encoder_scorer(model_name: str) -> Callable[[list[tuple[str, str]]], list[float]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        # bge-reranker has one output label, so CrossEncoder applies a sigmoid -> [0, 1].
        return [float(x) for x in model.predict(pairs, batch_size=16, show_progress_bar=False)]

    return score
