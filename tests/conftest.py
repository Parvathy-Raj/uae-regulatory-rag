from __future__ import annotations

import pytest

from uae_rag.schemas import Chunk, ScoredChunk


def make_chunk(
    cid: str, text: str = "some text", heading: str = "H", source: str = "doc", page: int = 1
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_id=source,
        source_title="Doc Title (2024)",
        domain="vat",
        url="https://example.org/doc.pdf",
        heading=heading,
        page_start=page,
        page_end=page,
        text=text,
    )


def scored(cid: str, score: float = 1.0, **kw) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(cid, **kw), score=score)


@pytest.fixture
def chunk_factory():
    return make_chunk


@pytest.fixture
def scored_factory():
    return scored
