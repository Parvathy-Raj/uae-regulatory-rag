"""Section-aware chunking with a word cap and overlap. Each chunk keeps its page range."""

from __future__ import annotations

from uae_rag.config import ChunkingConfig
from uae_rag.schemas import Chunk, Line, ParsedDoc, Section


def merge_small_sections(sections: list[Section], min_words: int) -> list[Section]:
    """Fold sections shorter than min_words into the following section."""
    merged: list[Section] = []
    carry: Section | None = None
    for section in sections:
        if carry is not None:
            section = Section(heading=carry.heading, lines=carry.lines + section.lines)
            carry = None
        if section.word_count() < min_words:
            carry = section
        else:
            merged.append(section)
    if carry is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Section(heading=last.heading, lines=last.lines + carry.lines)
        else:
            merged.append(carry)
    return merged


def _word_pages(lines: list[Line]) -> list[tuple[str, int]]:
    return [(word, line.page) for line in lines for word in line.text.split()]


def window_section(section: Section, max_words: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split a section into (text, page_start, page_end) windows."""
    words = _word_pages(section.lines)
    if not words:
        return []
    step = max(1, max_words - overlap)
    windows: list[tuple[str, int, int]] = []
    for start in range(0, len(words), step):
        part = words[start : start + max_words]
        windows.append((" ".join(w for w, _ in part), part[0][1], part[-1][1]))
        if start + max_words >= len(words):
            break
    return windows


def chunk_document(doc: ParsedDoc, cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = merge_small_sections(doc.sections, cfg.min_words)
    for s_idx, section in enumerate(sections):
        for c_idx, (text, p_start, p_end) in enumerate(
            window_section(section, cfg.max_words, cfg.overlap_words)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_id}:{s_idx:03d}:{c_idx:02d}",
                    source_id=doc.source_id,
                    source_title=doc.title,
                    domain=doc.domain,
                    url=doc.url,
                    heading=section.heading,
                    page_start=p_start,
                    page_end=p_end,
                    text=text,
                )
            )
    return chunks
