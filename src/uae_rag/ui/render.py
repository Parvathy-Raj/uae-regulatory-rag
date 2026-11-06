"""HTML rendering helpers for the chat UI (pure functions, unit-tested).

Model output is always HTML-escaped before any markup is added, so an answer can
never inject HTML or scripts into the page.
"""

from __future__ import annotations

import html
import re

from uae_rag.generate.answer import Answer

MODE_LABELS = {
    "dense": "Dense",
    "hybrid": "Hybrid (BM25 + dense)",
    "hybrid_rerank": "Hybrid + reranker",
}


def answer_html(text: str) -> str:
    """Escape, then add bold, line breaks and citation badges like [2] -> <sup>2</sup>."""
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"\[(\d{1,2})\]", r'<sup class="cite">\1</sup>', safe)
    return safe.replace("\n", "<br>")


def bubble(role: str, inner_html: str, extra_class: str = "") -> str:
    return f'<div class="msg {role} {extra_class}"><div class="bubble">{inner_html}</div></div>'


def page_label(start: int, end: int) -> str:
    return f"p.{start}" if start == end else f"pp.{start}–{end}"


def short_title(title: str, limit: int = 34) -> str:
    base = title.split("(")[0].strip()
    return base if len(base) <= limit else base[: limit - 1] + "…"


def sources_html(answer: Answer) -> str:
    """Chips for each retrieved passage the answer actually cited, in context order."""
    cited = set(answer.citations)
    chips = []
    for i, item in enumerate(answer.retrieved, start=1):
        c = item.chunk
        if c.chunk_id not in cited:
            continue
        href = html.escape(f"{c.url}#page={c.page_start}", quote=True)
        label = html.escape(
            f"{short_title(c.source_title)} · {page_label(c.page_start, c.page_end)}"
        )
        chips.append(
            f'<a class="src" href="{href}" target="_blank" rel="noopener"><b>{i}</b>{label}</a>'
        )
    return f'<div class="srcs">{"".join(chips)}</div>' if chips else ""


def meta_html(answer: Answer) -> str:
    total = sum(answer.latency_ms.values()) / 1000 if answer.latency_ms else 0.0
    parts = [MODE_LABELS.get(answer.mode or "", answer.mode or ""), f"{total:.1f}s"]
    if answer.refused:
        parts.append(f"refused: {answer.refusal_reason.replace('_', ' ')}")
    return f'<div class="meta">{html.escape(" · ".join(p for p in parts if p))}</div>'
