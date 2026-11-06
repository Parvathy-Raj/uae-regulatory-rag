"""Download the PDFs listed in corpus.yaml into data/raw/ (never committed)."""

from __future__ import annotations

from pathlib import Path

import httpx

from uae_rag.config import Corpus, CorpusSource

USER_AGENT = "uae-regulatory-rag/0.1 (research; corpus downloader)"


def target_path(raw_dir: Path, source: CorpusSource) -> Path:
    return raw_dir / f"{source.id}.pdf"


def is_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def download_source(
    source: CorpusSource, raw_dir: Path, client: httpx.Client, force: bool = False
) -> tuple[Path, str]:
    """Download one source. Returns (path, status) where status is 'cached' or 'downloaded'."""
    path = target_path(raw_dir, source)
    if path.exists() and not force:
        return path, "cached"
    resp = client.get(str(source.url))
    resp.raise_for_status()
    if not is_pdf(resp.content):
        raise ValueError(f"{source.id}: URL did not return a PDF ({source.url})")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return path, "downloaded"


def download_corpus(corpus: Corpus, raw_dir: Path, force: bool = False) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120
    ) as client:
        for source in corpus.sources:
            try:
                _, status = download_source(source, raw_dir, client, force)
            except (httpx.HTTPError, ValueError) as exc:
                status = f"FAILED: {exc}"
            results.append((source.id, status))
    return results
