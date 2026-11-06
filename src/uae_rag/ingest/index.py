"""Embed chunks and store them: Chroma for dense search, chunks.jsonl as the source of truth.

BM25 is rebuilt in memory from chunks.jsonl at load time (fast for a few thousand
chunks), so there is no second index to keep in sync.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from uae_rag.schemas import Chunk


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """bge-small-en-v1.5 on CPU. Queries get the bge retrieval instruction prefix."""

    def __init__(self, model_name: str, query_instruction: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(self.query_instruction + text, normalize_embeddings=True)
        return vector.tolist()


def write_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
            count += 1
    return count


def read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make ingest` first.")
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def open_collection(chroma_dir: Path, name: str, reset: bool = False):
    """Open (or create) the cosine-distance collection. We always pass our own vectors,
    so Chroma's built-in embedding function is never used for adds or queries."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    if reset:
        with contextlib.suppress(Exception):
            client.delete_collection(name)
    return client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})


def build_vector_index(
    chunks: list[Chunk], embedder: Embedder, chroma_dir: Path, name: str, batch: int = 128
) -> int:
    collection = open_collection(chroma_dir, name, reset=True)
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            documents=[c.text for c in part],
            metadatas=[c.metadata() for c in part],
            embeddings=embedder.embed_documents([c.embed_text() for c in part]),
        )
    return collection.count()


def to_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
