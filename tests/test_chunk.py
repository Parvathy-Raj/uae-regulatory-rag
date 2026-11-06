from uae_rag.config import ChunkingConfig
from uae_rag.ingest.chunk import chunk_document, merge_small_sections, window_section
from uae_rag.schemas import Line, ParsedDoc, Section


def section(heading: str, words: int, page: int = 1) -> Section:
    return Section(
        heading=heading, lines=[Line(page=page, text=" ".join(f"w{i}" for i in range(words)))]
    )


def test_window_respects_cap_and_overlap():
    s = section("H", 25)
    windows = window_section(s, max_words=10, overlap=3)
    assert all(len(text.split()) <= 10 for text, _, _ in windows)
    first, second = windows[0][0].split(), windows[1][0].split()
    assert first[-3:] == second[:3]
    assert windows[-1][0].split()[-1] == "w24"  # nothing lost at the end


def test_window_tracks_page_range():
    s = Section(heading="H", lines=[Line(page=4, text="a b c"), Line(page=5, text="d e f")])
    [(text, start, end)] = window_section(s, max_words=10, overlap=2)
    assert (start, end) == (4, 5)
    assert text == "a b c d e f"


def test_small_sections_merge_forward():
    merged = merge_small_sections([section("Tiny", 5), section("Big", 50)], min_words=20)
    assert len(merged) == 1
    assert merged[0].heading == "Tiny"
    assert merged[0].word_count() == 55


def test_trailing_small_section_merges_backward():
    merged = merge_small_sections([section("Big", 50), section("Tiny", 5)], min_words=20)
    assert len(merged) == 1 and merged[0].word_count() == 55


def test_chunk_ids_are_unique_and_carry_metadata():
    doc = ParsedDoc(
        source_id="ct_general",
        title="CT",
        domain="corporate_tax",
        url="https://x/y.pdf",
        num_pages=3,
        sections=[section("A", 120, page=2), section("B", 60, page=3)],
    )
    chunks = chunk_document(doc, ChunkingConfig(max_words=50, overlap_words=10, min_words=10))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "ct_general:000:00"
    assert chunks[0].metadata()["page"] == 2
    assert {c.heading for c in chunks} == {"A", "B"}
