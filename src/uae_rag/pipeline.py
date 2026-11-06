"""Wires the store, retrievers and generator together. Models load lazily and once."""

from __future__ import annotations

import time
from functools import cached_property

from uae_rag.config import RetrievalMode, Settings
from uae_rag.generate.answer import Answer, generate_answer
from uae_rag.ingest.index import SentenceTransformerEmbedder, open_collection, read_chunks
from uae_rag.llm.client import LLMClient
from uae_rag.retrieve.base import Retriever
from uae_rag.retrieve.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    cross_encoder_scorer,
)
from uae_rag.schemas import Chunk, ScoredChunk


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def chunks(self) -> list[Chunk]:
        return read_chunks(self.settings.paths.chunks_file)

    @cached_property
    def by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    @cached_property
    def embedder(self) -> SentenceTransformerEmbedder:
        e = self.settings.embedding
        return SentenceTransformerEmbedder(e.model, e.query_instruction, e.batch_size)

    @cached_property
    def dense(self) -> DenseRetriever:
        p = self.settings.paths
        return DenseRetriever(
            open_collection(p.chroma_dir, p.collection), self.embedder.embed_query, self.by_id
        )

    @cached_property
    def bm25(self) -> BM25Retriever:
        return BM25Retriever(self.chunks)

    @cached_property
    def hybrid(self) -> HybridRetriever:
        r = self.settings.retrieval
        return HybridRetriever(self.dense, self.bm25, r.candidate_k, r.rrf_k)

    @cached_property
    def reranked(self) -> RerankRetriever:
        r = self.settings.retrieval
        return RerankRetriever(
            self.hybrid, cross_encoder_scorer(self.settings.reranker.model), r.candidate_k
        )

    @cached_property
    def llm(self) -> LLMClient:
        return LLMClient(self.settings.llm)

    def retriever(self, mode: RetrievalMode) -> Retriever:
        return {"dense": self.dense, "hybrid": self.hybrid, "hybrid_rerank": self.reranked}[mode]

    def retrieve(self, question: str, mode: RetrievalMode | None = None) -> list[ScoredChunk]:
        mode = mode or self.settings.retrieval.mode
        return self.retriever(mode).retrieve(question, self.settings.retrieval.top_k)

    def ask(self, question: str, mode: RetrievalMode | None = None) -> Answer:
        mode = mode or self.settings.retrieval.mode
        t0 = time.perf_counter()
        chunks = self.retrieve(question, mode)
        t1 = time.perf_counter()
        answer = generate_answer(question, chunks, self.llm, self.settings.generation, mode)
        t2 = time.perf_counter()
        answer.latency_ms = {"retrieve": (t1 - t0) * 1000, "generate": (t2 - t1) * 1000}
        return answer
