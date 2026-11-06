"""PDF -> lines (with page + font info) -> sections split on headings.

Heading detection is heuristic: a line is a heading when its font is clearly larger
than the body text, or when it is bold, short and looks like a numbered heading
("4.2 Exempt supplies"). Running headers/footers, bare page numbers and
table-of-contents entries are dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from uae_rag.config import CorpusSource
from uae_rag.schemas import Line, ParsedDoc, Section

NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+\S")
PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
# Table-of-contents entries: "4. Requirements to be a QFZP ........ 21"
TOC_LINE = re.compile(r"(\.\s?){4,}\s*\d*\s*$|(\.\s?){6,}")


def drop_toc_lines(lines: list[Line]) -> list[Line]:
    """Remove table-of-contents entries; they duplicate headings and confuse retrieval."""
    return [ln for ln in lines if not TOC_LINE.search(ln.text)]


MAX_HEADING_WORDS = 14
MAX_UNNUMBERED_HEADING_WORDS = 8
SIZE_RATIO = 1.15
SMALL_PRINT_RATIO = 0.88


def _normalise(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def body_font_size(lines: list[Line]) -> float:
    """Most common font size, weighted by words: that is the body text."""
    counts: Counter[float] = Counter()
    for line in lines:
        counts[round(line.size, 1)] += len(line.text.split())
    return counts.most_common(1)[0][0] if counts else 0.0


def _edge_flags(lines: list[Line]) -> list[bool]:
    """True for the first and last line of each page (pages with at least 4 lines)."""
    by_page: dict[int, list[int]] = {}
    for i, line in enumerate(lines):
        by_page.setdefault(line.page, []).append(i)
    flags = [False] * len(lines)
    for idxs in by_page.values():
        if len(idxs) >= 4:
            flags[idxs[0]] = flags[idxs[-1]] = True
    return flags


def drop_repeated_lines(lines: list[Line], min_share: float = 0.5) -> list[Line]:
    """Remove running headers/footers and bare page numbers.

    A line is a running header/footer when it appears on >= min_share of pages either
    verbatim, or with only its numbers changing (e.g. "VAT Guide | Page 7") while sitting
    at the top or bottom edge of the page. The edge rule keeps body lines such as
    "Article 3" that merely look alike once digits are masked.
    """
    is_page_no = [bool(PAGE_NUMBER.match(ln.text.strip())) for ln in lines]
    pages = {line.page for line in lines}
    if len(pages) < 3:
        return [ln for ln, pn in zip(lines, is_page_no, strict=True) if not pn]
    limit = max(2, int(len(pages) * min_share))
    edges = _edge_flags(lines)
    exact: dict[str, set[int]] = {}
    masked: dict[str, set[int]] = {}
    for line, edge in zip(lines, edges, strict=True):
        exact.setdefault(line.text.strip(), set()).add(line.page)
        if edge:
            masked.setdefault(_normalise(line.text), set()).add(line.page)
    keep = []
    for line, edge, pn in zip(lines, edges, is_page_no, strict=True):
        repeated = len(exact[line.text.strip()]) >= limit or (
            edge and len(masked.get(_normalise(line.text), ())) >= limit
        )
        if not (repeated or pn):
            keep.append(line)
    return keep


def is_heading(line: Line, body_size: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not words or len(words) > MAX_HEADING_WORDS or text.endswith((".", ",", ";")):
        return False
    if not any(c.isalpha() for c in text):
        return False
    if body_size and line.size >= body_size * SIZE_RATIO:
        return True
    if not line.bold:
        return False
    # Bold numbered line ("4.2 Exempt supplies"), or a short fully-bold line whose
    # number was drawn separately by the PDF ("Free Zone Persons").
    return bool(NUMBERED_HEADING.match(text)) or (
        len(words) <= MAX_UNNUMBERED_HEADING_WORDS and text[0].isupper()
    )


def drop_small_print(lines: list[Line], body_size: float) -> list[Line]:
    """Drop footnotes and superscript markers: lines clearly smaller than body text.
    pdfplumber reads them in page order, so they would split body sentences."""
    if not body_size:
        return lines
    return [ln for ln in lines if not (0 < ln.size < body_size * SMALL_PRINT_RATIO)]


def split_sections(lines: list[Line], body_size: float, default_heading: str) -> list[Section]:
    sections: list[Section] = [Section(heading=default_heading)]
    for line in lines:
        if is_heading(line, body_size):
            current = sections[-1]
            text = line.text.strip()
            # A heading wrapped over two lines is joined, but a new numbered heading
            # ("2.1. Overview" right after "2. Introduction") starts its own section.
            wrapped = not NUMBERED_HEADING.match(text)
            if current.lines == [] and current.heading != default_heading and wrapped:
                current.heading = f"{current.heading} {text}"
            else:
                sections.append(Section(heading=text))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if s.lines]


def _page_lines(page: object, page_no: int) -> list[Line]:
    words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)  # type: ignore[attr-defined]
    rows: dict[int, list[dict]] = {}
    for word in words:
        rows.setdefault(round(word["top"] / 3), []).append(word)
    lines: list[Line] = []
    for key in sorted(rows):
        row = sorted(rows[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in row).strip()
        if not text:
            continue
        size = max(w["size"] for w in row)
        bold = all("bold" in str(w["fontname"]).lower() for w in row)
        lines.append(Line(page=page_no, text=text, size=size, bold=bold))
    return lines


def read_pdf_lines(path: Path) -> tuple[list[Line], int]:
    import pdfplumber

    lines: list[Line] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            lines.extend(_page_lines(page, i))
        return lines, len(pdf.pages)


def parse_pdf(path: Path, source: CorpusSource) -> ParsedDoc:
    raw_lines, num_pages = read_pdf_lines(path)
    lines = drop_toc_lines(drop_repeated_lines(raw_lines))
    body = body_font_size(lines)
    sections = split_sections(drop_small_print(lines, body), body, default_heading=source.title)
    return ParsedDoc(
        source_id=source.id,
        title=source.title,
        domain=source.domain,
        url=str(source.url),
        num_pages=num_pages,
        sections=sections,
    )
