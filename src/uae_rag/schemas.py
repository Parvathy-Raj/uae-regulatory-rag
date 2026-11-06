"""Core data models shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Line(BaseModel):
    """One line of text from a PDF page, with the font info used to spot headings."""

    page: int
    text: str
    size: float = 0.0
    bold: bool = False


class Section(BaseModel):
    heading: str
    lines: list[Line] = Field(default_factory=list)

    @property
    def page_start(self) -> int:
        return self.lines[0].page if self.lines else 0

    def word_count(self) -> int:
        return sum(len(line.text.split()) for line in self.lines)


class ParsedDoc(BaseModel):
    source_id: str
    title: str
    domain: str
    url: str
    num_pages: int
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    source_title: str
    domain: str
    url: str
    heading: str
    page_start: int
    page_end: int
    text: str

    def embed_text(self) -> str:
        """Text used for embedding and BM25: heading gives the chunk its topic."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text

    def metadata(self) -> dict[str, str | int]:
        """Flat metadata for the vector store (scalar values only)."""
        return {
            "source": self.source_id,
            "source_title": self.source_title,
            "domain": self.domain,
            "url": self.url,
            "heading": self.heading,
            "page": self.page_start,
            "page_end": self.page_end,
            "chunk_id": self.chunk_id,
        }


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    stage_scores: dict[str, float] = Field(default_factory=dict)
