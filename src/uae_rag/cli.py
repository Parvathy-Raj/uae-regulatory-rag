"""Command line entry point: `python -m uae_rag.cli <command>` (or `uae-rag <command>`)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from uae_rag.config import RETRIEVAL_MODES, ROOT, RetrievalMode, load_corpus, load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="UAE regulatory RAG (Stage 1).")
console = Console()


def _modes(value: str) -> list[RetrievalMode]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    bad = [m for m in modes if m not in RETRIEVAL_MODES]
    if bad:
        raise typer.BadParameter(f"Unknown mode(s) {bad}. Choose from {list(RETRIEVAL_MODES)}")
    return modes  # type: ignore[return-value]


# --------------------------------------------------------------------------- corpus
@app.command()
def download(force: bool = False) -> None:
    """Download the PDFs listed in corpus.yaml into data/raw/."""
    from uae_rag.ingest.download import download_corpus

    s = load_settings()
    results = download_corpus(load_corpus(s.paths.corpus_file), s.paths.raw_dir, force)
    for source_id, status in results:
        console.print(f"  {source_id:<28} {status}")
    if any(st.startswith("FAILED") for _, st in results):
        raise typer.Exit(1)


@app.command()
def ingest() -> None:
    """Parse PDFs -> sections -> chunks, then embed into Chroma."""
    from uae_rag.ingest.chunk import chunk_document
    from uae_rag.ingest.download import target_path
    from uae_rag.ingest.index import write_chunks
    from uae_rag.ingest.parse import parse_pdf

    s = load_settings()
    corpus = load_corpus(s.paths.corpus_file)
    all_chunks = []
    s.paths.parsed_dir.mkdir(parents=True, exist_ok=True)
    for source in corpus.sources:
        pdf = target_path(s.paths.raw_dir, source)
        if not pdf.exists():
            console.print(f"[red]Missing {pdf}. Run `make download` first.[/red]")
            raise typer.Exit(1)
        doc = parse_pdf(pdf, source)
        (s.paths.parsed_dir / f"{source.id}.json").write_text(
            doc.model_dump_json(indent=1), encoding="utf-8"
        )
        chunks = chunk_document(doc, s.chunking)
        all_chunks.extend(chunks)
        console.print(
            f"  {source.id:<28} {doc.num_pages:>4} pages  {len(doc.sections):>4} sections  {len(chunks):>5} chunks"
        )
    write_chunks(all_chunks, s.paths.chunks_file)
    _embed_and_store(all_chunks)


@app.command()
def index() -> None:
    """Re-embed data/chunks.jsonl into Chroma without re-parsing the PDFs."""
    from uae_rag.ingest.index import read_chunks

    s = load_settings()
    _embed_and_store(read_chunks(s.paths.chunks_file))


def _embed_and_store(chunks: list) -> None:
    from uae_rag.ingest.index import SentenceTransformerEmbedder, build_vector_index

    s = load_settings()
    embedder = SentenceTransformerEmbedder(
        s.embedding.model, s.embedding.query_instruction, s.embedding.batch_size
    )
    count = build_vector_index(chunks, embedder, s.paths.chroma_dir, s.paths.collection)
    console.print(f"[green]Indexed {count} chunks.[/green]")


# --------------------------------------------------------------------------- ask
@app.command()
def ask(
    question: str,
    mode: Annotated[str, typer.Option(help="dense | hybrid | hybrid_rerank")] = "",
    show_context: bool = False,
) -> None:
    """Answer one question with citations."""
    from uae_rag.pipeline import RAGPipeline

    s = load_settings()
    pipe = RAGPipeline(s)
    ans = pipe.ask(question, _modes(mode)[0] if mode else None)
    style = "yellow" if ans.refused else "green"
    console.print(Panel(ans.answer, title=f"{ans.mode} · {ans.model}", border_style=style))
    for i, item in enumerate(ans.retrieved, start=1):
        c = item.chunk
        mark = "★" if c.chunk_id in ans.citations else " "
        console.print(
            f" {mark} [{i}] {c.chunk_id}  p.{c.page_start}  {c.heading[:60]}  (score {item.score:.3f})"
        )
        if show_context:
            console.print(f"     {c.text[:400]}…", style="dim")
    console.print(f"latency ms: { {k: round(v) for k, v in ans.latency_ms.items()} }", style="dim")


# --------------------------------------------------------------------------- golden set
@app.command("draft-golden")
def draft_golden(n: int = 120, seed: int = 7) -> None:
    """Draft n questions with the LLM into eval/golden.jsonl (status=pending)."""
    from uae_rag.evaluation.golden import (
        DRAFT_PROMPTS,
        DraftQA,
        draft_items,
        merge_drafts,
        read_golden,
        write_golden,
    )
    from uae_rag.pipeline import RAGPipeline

    s = load_settings()
    pipe = RAGPipeline(s)

    def neighbours(chunk):
        return [r.chunk for r in pipe.dense.retrieve(chunk.embed_text()[:1000], 15)]

    def draft(category, chunks):
        passages = "\n\n".join(
            f"PASSAGE {i}:\n{c.heading}\n{c.text}" for i, c in enumerate(chunks, 1)
        )
        return pipe.llm.structured(
            [
                {"role": "system", "content": DRAFT_PROMPTS[category]},
                {"role": "user", "content": passages},
            ],
            DraftQA,
        )

    with console.status("Drafting questions…"):
        items = draft_items(pipe.chunks, n, draft, neighbours, seed=seed, model_name=s.llm.model)
    merged = merge_drafts(read_golden(s.paths.golden_file), items)
    write_golden(merged, s.paths.golden_file)
    console.print(
        f"[green]{len(items)} drafts added. {len(merged)} items in {s.paths.golden_file}.[/green] "
        "Now run `make verify-golden`."
    )


@app.command("verify-golden")
def verify_golden() -> None:
    """Review pending items one by one: [a]pprove, [e]dit, [r]eject, [s]kip, [q]uit."""
    from uae_rag.evaluation.golden import apply_decision, read_golden, write_golden
    from uae_rag.ingest.index import read_chunks

    s = load_settings()
    items = read_golden(s.paths.golden_file)
    by_id = {c.chunk_id: c for c in read_chunks(s.paths.chunks_file)}
    pending = [i for i, it in enumerate(items) if it.status == "pending"]
    console.print(
        f"{len(pending)} pending · {sum(it.status == 'approved' for it in items)} approved"
    )
    for pos, idx in enumerate(pending, start=1):
        item = items[idx]
        console.rule(f"{pos}/{len(pending)} · {item.category} · {item.id}")
        seeds = item.supporting_chunk_ids or [item.notes.removeprefix("seed_chunk=")]
        for cid in seeds:
            c = by_id.get(cid)
            if c:
                console.print(
                    Panel(
                        c.text,
                        title=f"{cid} · p.{c.page_start} · {c.heading[:70]}",
                        border_style="dim",
                    )
                )
        console.print(f"[bold]Q:[/bold] {item.question}\n[bold]A:[/bold] {item.expected_answer}")
        choice = Prompt.ask(
            "approve / edit / reject / skip / quit", choices=["a", "e", "r", "s", "q"], default="a"
        )
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "e":
            edits = {
                "question": Prompt.ask("question", default=item.question),
                "expected_answer": Prompt.ask("expected_answer", default=item.expected_answer),
                "category": Prompt.ask(
                    "category",
                    choices=["easy", "multi_section", "unanswerable"],
                    default=item.category,
                ),
                "supporting_chunk_ids": [
                    x.strip()
                    for x in Prompt.ask(
                        "supporting_chunk_ids (comma separated)",
                        default=",".join(item.supporting_chunk_ids),
                    ).split(",")
                    if x.strip()
                ],
            }
            items[idx] = apply_decision(item, "edit", edits)
        else:
            items[idx] = apply_decision(item, "approve" if choice == "a" else "reject")
        write_golden(items, s.paths.golden_file)  # save after every decision
    console.print(
        f"[green]Saved. {sum(it.status == 'approved' for it in items)} approved items.[/green]"
    )


@app.command("ci-subset")
def ci_subset(n: int = 15, seed: int = 13) -> None:
    """Write a small stratified subset of approved items to eval/golden_ci.jsonl."""
    from uae_rag.evaluation.golden import approved, read_golden, stratified_subset, write_golden

    s = load_settings()
    items = approved(read_golden(s.paths.golden_file))
    if not items:
        console.print("[red]No approved items yet.[/red]")
        raise typer.Exit(1)
    subset = stratified_subset(items, n, seed)
    write_golden(subset, s.paths.golden_ci_file)
    console.print(f"[green]{len(subset)} items -> {s.paths.golden_ci_file}[/green]")


# --------------------------------------------------------------------------- eval
@app.command("eval")
def run_eval(
    golden: Annotated[Path | None, typer.Option(help="Defaults to eval/golden.jsonl")] = None,
    modes: str = ",".join(RETRIEVAL_MODES),
    limit: int = 0,
    generate: bool = True,
    judge: bool = True,
    track: bool = True,
    save_baseline: bool = False,
    tag: str = "",
) -> None:
    """Evaluate approved golden items for each retrieval mode; log to MLflow."""
    from uae_rag.evaluation.gate import report_to_plain
    from uae_rag.evaluation.golden import approved, dataset_hash, read_golden
    from uae_rag.evaluation.report import category_table, mode_comparison_table
    from uae_rag.evaluation.runner import run_mode, write_mode_run
    from uae_rag.evaluation.tracking import log_run
    from uae_rag.llm.client import LLMClient
    from uae_rag.pipeline import RAGPipeline

    s = load_settings()
    path = golden or s.paths.golden_file
    items = approved(read_golden(path))
    if limit:
        items = items[:limit]
    if not items:
        console.print(
            f"[red]No approved items in {path}. Run draft-golden + verify-golden first.[/red]"
        )
        raise typer.Exit(1)
    pipe = RAGPipeline(s)
    gen_llm = pipe.llm if generate else None
    judge_llm = LLMClient(s.llm, model=s.llm.judge_model) if (generate and judge) else None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = s.paths.runs_dir / f"{stamp}{'-' + tag if tag else ''}"
    ds = dataset_hash(items)
    reports = {}
    for mode in _modes(modes):
        with console.status(f"Evaluating {mode} on {len(items)} items…") as status:
            run = run_mode(
                pipe, items, mode, gen_llm, judge_llm, progress=lambda m: status.update(m)
            )
        write_mode_run(run, out_dir)
        reports[mode] = run.report
        console.print(Markdown(f"### {mode}\n\n" + category_table(run.report)))
        errors = sum(1 for r in run.results if r.error)
        if errors:
            console.print(f"[yellow]{errors} items had LLM errors; see {out_dir}[/yellow]")
        if track:
            log_run(
                s,
                mode,
                run.report,
                out_dir,
                {"dataset": path.name, "dataset_hash": ds, "n_items": str(len(items))},
                ROOT,
            )
    comparison = mode_comparison_table(reports)
    (out_dir / "results.md").write_text(
        f"Dataset: {path.name} ({len(items)} items, hash {ds}) · model {s.llm.model} · judge {s.llm.judge_model}\n\n"
        + comparison
        + "\n\n"
        + "\n\n".join(f"#### {m}\n\n{category_table(r)}" for m, r in reports.items()),
        encoding="utf-8",
    )
    console.print(Markdown("### Retriever comparison\n\n" + comparison))
    console.print(f"Run saved to {out_dir}")
    if save_baseline:
        from uae_rag.evaluation.gate import load_gate_config

        gate_mode = load_gate_config(ROOT / "config" / "gate.yaml").mode
        if gate_mode not in reports:
            console.print(f"[red]Baseline needs mode {gate_mode} in --modes.[/red]")
            raise typer.Exit(1)
        payload = {
            "mode": gate_mode,
            "dataset_hash": ds,
            "created": stamp,
            "model": s.llm.model,
            "judge_model": s.llm.judge_model,
            "report": report_to_plain(reports[gate_mode]),
        }
        s.paths.baseline_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Baseline written to {s.paths.baseline_file}[/green]")


@app.command()
def gate(
    run_dir: Path,
    comment_out: Annotated[
        Path | None, typer.Option(help="Write the PR comment markdown here")
    ] = None,
) -> None:
    """Compare a run against eval/baseline.json. Exits 1 when the gate fails."""
    from uae_rag.evaluation.gate import evaluate_gate, gate_markdown, load_gate_config, read_json
    from uae_rag.evaluation.metrics import MetricSummary
    from uae_rag.evaluation.report import category_table

    s = load_settings()
    cfg = load_gate_config(ROOT / "config" / "gate.yaml")
    current_raw = read_json(run_dir / f"report_{cfg.mode}.json")
    if current_raw is None:
        console.print(f"[red]No report_{cfg.mode}.json in {run_dir}[/red]")
        raise typer.Exit(2)
    current = current_raw["report"]
    baseline_raw = read_json(s.paths.baseline_file)
    result = evaluate_gate(current, baseline_raw["report"] if baseline_raw else None, cfg)
    table = category_table(
        {k: {m: MetricSummary(**v) for m, v in ms.items()} for k, ms in current.items()}
    )
    md = gate_markdown(
        result, table, header=f"### Eval gate · `{cfg.mode}` · model `{s.llm.model}`"
    )
    if comment_out:
        comment_out.write_text(md, encoding="utf-8")
    console.print(Markdown(md))
    if not result.passed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- calibration
@app.command("calibrate-label")
def calibrate_label(items_file: Path, n: int = 30) -> None:
    """Label judge-scored answers yourself (from an eval run's items_<mode>.jsonl)."""
    from uae_rag.evaluation.calibration import HumanLabel, read_results, sample_for_labeling

    out = ROOT / "eval" / "calibration" / "human_labels.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {
            json.loads(x)["item_id"]
            for x in out.read_text(encoding="utf-8").splitlines()
            if x.strip()
        }
        if out.exists()
        else set()
    )
    mode = items_file.stem.removeprefix("items_")
    sample = [r for r in sample_for_labeling(read_results(items_file), n) if r.item_id not in done]
    console.print(f"{len(sample)} answers to label (judge scores are hidden on purpose).")
    for i, r in enumerate(sample, start=1):
        console.rule(f"{i}/{len(sample)} · {r.category}")
        for j, ctx in enumerate(r.context, start=1):
            mark = "★" if r.retrieved_ids[j - 1] in r.cited_ids else " "
            console.print(
                Panel(ctx, title=f"{mark}[{j}] {r.retrieved_ids[j - 1]}", border_style="dim")
            )
        console.print(
            f"[bold]Q:[/bold] {r.question}\n[bold]Reference:[/bold] {r.expected_answer}\n[bold]Answer:[/bold] {r.answer}"
        )
        faith = Prompt.ask("Every claim supported by the passages?", choices=["y", "n", "q"])
        if faith == "q":
            break
        corr = None
        if r.category != "unanswerable":
            c = Prompt.ask(
                "Correct vs reference? [c]orrect [p]artial [i]ncorrect", choices=["c", "p", "i"]
            )
            corr = {"c": "correct", "p": "partial", "i": "incorrect"}[c]
        label = HumanLabel(
            item_id=r.item_id,
            mode=mode,
            faithfulness="faithful" if faith == "y" else "unfaithful",
            correctness=corr,
        )
        with out.open("a", encoding="utf-8") as fh:
            fh.write(label.model_dump_json() + "\n")
    console.print(f"[green]Labels saved to {out}[/green]")


@app.command("calibrate-report")
def calibrate_report(items_file: Path) -> None:
    """Agreement and Cohen's kappa between your labels and the LLM judge."""
    from uae_rag.evaluation.calibration import HumanLabel, compare, read_results

    path = ROOT / "eval" / "calibration" / "human_labels.jsonl"
    if not path.exists():
        console.print("[red]No human labels yet. Run calibrate-label first.[/red]")
        raise typer.Exit(1)
    labels = [
        HumanLabel.model_validate_json(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    rows = [
        "| Metric | n | Agreement | Cohen's κ | Confusion (human→judge) |",
        "|---|---|---|---|---|",
    ]
    for a in compare(labels, read_results(items_file)):
        agree = "–" if a.agreement is None else f"{a.agreement:.2f}"
        kappa = "–" if a.kappa is None else f"{a.kappa:.2f}"
        conf = ", ".join(f"{k}: {v}" for k, v in a.confusion.items())
        rows.append(f"| {a.metric} | {a.n} | {agree} | {kappa} | {conf} |")
    md = "\n".join(rows)
    (path.parent / "agreement.md").write_text(md + "\n", encoding="utf-8")
    console.print(Markdown(md))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
