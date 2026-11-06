from uae_rag.ingest.parse import body_font_size, drop_repeated_lines, is_heading, split_sections
from uae_rag.schemas import Line


def L(text: str, page: int = 1, size: float = 10.0, bold: bool = False) -> Line:
    return Line(page=page, text=text, size=size, bold=bold)


def test_body_font_size_is_most_common_by_words():
    lines = [L("a b c d e f", size=10), L("Big Title", size=16), L("g h i j", size=10)]
    assert body_font_size(lines) == 10.0


def test_large_font_line_is_heading():
    assert is_heading(L("Registration for VAT", size=14), body_size=10)


def test_bold_numbered_line_is_heading_but_bold_sentence_is_not():
    assert is_heading(L("4.2 Exempt supplies", bold=True), body_size=10)
    assert not is_heading(L("Note that this applies to all persons.", bold=True), body_size=10)


def test_long_or_punctuated_lines_are_not_headings():
    assert not is_heading(L("This is a body sentence that ends with a full stop.", size=14), 10)
    assert not is_heading(L(" ".join(["word"] * 20), size=14), 10)
    assert not is_heading(L("2024", size=14), 10)


def test_split_sections_tracks_headings_and_pages():
    lines = [
        L("intro text here", page=1),
        L("1. Overview", page=1, size=14),
        L("overview body", page=1),
        L("more overview", page=2),
        L("2. Rates", page=3, size=14),
        L("rates body", page=3),
    ]
    sections = split_sections(lines, body_size=10, default_heading="Doc")
    assert [s.heading for s in sections] == ["Doc", "1. Overview", "2. Rates"]
    assert sections[1].page_start == 1
    assert {ln.page for ln in sections[1].lines} == {1, 2}


def test_wrapped_heading_lines_are_joined():
    lines = [L("3. Free Zone", size=14), L("Persons", size=14), L("body", page=1)]
    sections = split_sections(lines, body_size=10, default_heading="Doc")
    assert sections[0].heading == "3. Free Zone Persons"


def test_running_headers_and_page_numbers_dropped():
    lines = []
    for p in range(1, 6):
        lines += [
            L("Federal Tax Authority | VAT Guide", page=p),
            L(f"content {p}", page=p),
            L(str(p), page=p),
        ]
    kept = drop_repeated_lines(lines)
    assert [ln.text for ln in kept] == [f"content {p}" for p in range(1, 6)]


def test_numbered_footer_dropped_but_similar_body_lines_kept():
    lines = []
    for p in range(1, 6):
        body = [L(f"Article {p} sets out rule {p}.", page=p)] + [
            L(f"body line {i}", page=p) for i in range(8)
        ]
        lines += [L("VAT Guide", page=p), *body, L(f"VAT Guide | Page {p} of 40", page=p)]
    texts = [ln.text for ln in drop_repeated_lines(lines)]
    assert not any(t.startswith("VAT Guide") for t in texts)
    assert "Article 3 sets out rule 3." in texts


def test_toc_lines_are_dropped_but_numbered_text_kept():
    from uae_rag.ingest.parse import drop_toc_lines

    lines = [
        L("4. Requirements to be a QFZP ............................ 21"),
        L("13. Updates and Amendments . . . . . . . . 134"),
        L("Section 3.2.1.1. applies here."),
        L("See e.g. Article 5."),
    ]
    assert [ln.text for ln in drop_toc_lines(lines)] == [
        "Section 3.2.1.1. applies here.",
        "See e.g. Article 5.",
    ]


def test_numbered_child_heading_starts_new_section():
    lines = [L("2. Introduction", size=14), L("2.1. Overview", size=14), L("body text")]
    sections = split_sections(lines, body_size=10, default_heading="Doc")
    assert [s.heading for s in sections] == ["2.1. Overview"]


def test_short_fully_bold_line_is_heading_even_without_number():
    assert is_heading(L("Free Zone Persons", bold=True), body_size=12)
    assert not is_heading(L("lowercase bold phrase here", bold=True), body_size=12)
    assert not is_heading(L("Free Zone Persons", bold=False), body_size=12)


def test_footnotes_in_small_print_are_dropped():
    from uae_rag.ingest.parse import drop_small_print

    lines = [
        L("Body sentence starts", size=12),
        L("39 Article 2(2) of Decision", size=9.96),
        L("41", size=8),
        L("and continues here.", size=12),
    ]
    assert [ln.text for ln in drop_small_print(lines, 12)] == [
        "Body sentence starts",
        "and continues here.",
    ]
