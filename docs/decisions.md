# Architecture decision records

Short records of the main trade-offs. Newest at the bottom.

## ADR-001: Groq (free tier) as default LLM, Ollama as offline option
- **Context:** The original brief asked for local Qwen2.5 via Ollama. The dev machine has 7.6 GB RAM and no GPU, where a 7B model is slow and leaves little memory for embeddings and the reranker.
- **Decision:** One OpenAI-compatible client. Default provider is Groq (`openai/gpt-oss-20b` to generate, `openai/gpt-oss-120b` to judge). Ollama with `qwen2.5:3b-instruct` is the keyless offline option.
- **Trade-off:** Groq needs a free API key and is rate-limited, so it's no longer "no API keys". Results depend on the provider, so the baseline records the model names.

## ADR-002: JSON mode + Pydantic validation instead of provider schema modes
- **Decision:** Ask for `response_format: json_object`, include the JSON schema in the prompt, validate with Pydantic, and send the validation error back once for a retry.
- **Why:** Works the same on Groq and Ollama, and with any model. Strict schema modes differ per provider and per model.
- **Found in testing:** gpt-oss is a reasoning model, and its hidden reasoning shares `max_tokens` with the answer. With 1024 tokens it sometimes returned an empty reply, which Groq rejects as HTTP 400 `json_validate_failed`. Fixes: `reasoning_effort: low` (the same setting Prompt Heist uses), `max_tokens: 2048`, and treating that 400 as a retryable bad-JSON reply.

## ADR-003: pdfplumber for parsing
- **Decision:** pdfplumber (MIT) with font-size and bold heuristics for headings. A heading is a larger-font line, a bold numbered line, or a short fully-bold line (some FTA guides draw the "5.5." number separately from the text). The parser also drops running headers/footers, table-of-contents lines, and small-print footnotes.
- **Trade-off:** Slower than PyMuPDF, but PyMuPDF is AGPL. Parsing 7 PDFs (about 650 pages) takes about 3 minutes, once.
- **Found on the real corpus:** contents entries were swallowing whole sections, and footnotes were splitting body sentences. Both are now fixed and tested. The cost of dropping footnotes: their legal references (e.g. "Article 66 of the Corporate Tax Law") are no longer searchable. Bold table header rows still become small sections of their own.

## ADR-004: Section-aware chunks, 300 words, 50 overlap
- **Decision:** Split on detected headings first, then window long sections. Sections under 40 words merge forward so headings don't become empty chunks.
- **Trade-off:** Word counts are a rough proxy for tokens. 300 words stays under bge-small's 512-token limit for typical English prose.

## ADR-005: BM25 rebuilt in memory, Chroma for vectors
- **Decision:** `chunks.jsonl` is the source of truth. BM25 is rebuilt at load time, and Chroma stores only vectors plus metadata.
- **Why:** Rebuilding BM25 takes seconds for a few thousand chunks, and it removes a second index that could drift out of sync.
- **Chroma details:** the collection is created with `configuration={"hnsw": {"space": "cosine"}}` (Chroma 1.x API; the older `metadata={"hnsw:space": ...}` still works in 1.5.9, but it's the legacy form). Reported distance was checked to equal 1 − dot product of the normalised bge vectors. We always pass our own embeddings, so Chroma's default MiniLM embedder is never used. Telemetry is off. `make index` re-embeds `chunks.jsonl` without re-parsing.

## ADR-006: Reciprocal rank fusion (k = 60) for hybrid
- **Decision:** RRF on ranks rather than mixing raw scores.
- **Why:** BM25 and cosine scores sit on different scales. RRF needs no tuning or normalisation, and k = 60 is the standard value from the original paper.

## ADR-007: Numeric citation labels mapped to chunk ids
- **Decision:** The LLM sees passages as `[1]..[5]` and cites those labels. Code maps them back to `chunk_id`s and drops labels that don't exist.
- **Why:** Small and mid-size models copy long ids unreliably. Invented citations are detected instead of trusted.

## ADR-008: Two-layer refusal
- **Decision:** (1) Refuse before calling the LLM when the top reranker score is below `min_rerank_score`. (2) Refuse when the model says the context is insufficient, or when it gives no valid citation.
- **Trade-off:** The citation rule trades some false refusals for fewer unsupported answers. Both rates are reported, as refusal accuracy and false refusal.

## ADR-009: Metric definitions
- recall@5: share of gold supporting chunks in the top 5. MRR: 1 / rank of the first gold chunk. Both apply to answerable items only.
- Faithfulness: the judge lists claims; the score is supported / total, computed in code.
- Correctness: judge verdict correct / partial / incorrect → 1 / 0.5 / 0. A refusal on an answerable question scores 0.
- Citation accuracy: answered items where at least one citation is a gold chunk.
- Refusal accuracy (unanswerable items) and false refusal (answerable items) are reported separately.
- Every metric is reported per category with its item count, so a good global average can't hide a weak slice.

## ADR-010: Judge uses a different, larger model
- **Decision:** Generate with gpt-oss-20b and judge with gpt-oss-120b, then calibrate the judge against 30 human labels (agreement + Cohen's κ).
- **Why:** This reduces self-preference bias. Calibration says how far to trust it.

## ADR-011: CI gate on a small subset
- **Decision:** CI runs about 15 stratified approved items on `hybrid_rerank` only. It fails if recall@5 or faithfulness drops by more than 0.05 (configurable), globally or in any slice with at least 3 items.
- **Trade-off:** A small subset is noisy, so the gate catches large regressions, not 1-point changes. Full evals run locally and are tracked in MLflow.

## ADR-012: MLflow with a local SQLite store
- **Decision:** `sqlite:///mlruns.db`, no tracking server. Each run logs flattened config, per-slice metrics, the git SHA (with `-dirty` when there are uncommitted changes), a dataset hash, and the run files as artifacts.

## ADR-013: Streamlit UI styled after Prompt Heist
- **Decision:** A Streamlit chat with custom CSS (same palette and layout as the Prompt Heist arena). Model output is HTML-escaped before any markup is added.
- **Why:** It gives a fast demo for the README GIF, and the escaping means a model answer can't inject HTML into the page.
