"""Markdown tables for eval results (used by the CLI, README and PR comments)."""

from __future__ import annotations

from collections.abc import Mapping

from uae_rag.evaluation.metrics import METRICS, Report

LABELS = {
    "recall_at_5": "Recall@5",
    "mrr": "MRR",
    "faithfulness": "Faithfulness",
    "answer_correctness": "Correctness",
    "refusal_accuracy": "Refusal acc.",
    "false_refusal": "False refusal ↓",
    "citation_accuracy": "Citation acc.",
}


def fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def category_table(report: Report) -> str:
    """Rows = slices (all, easy, ...), columns = metrics. Cells show mean (n)."""
    header = "| Slice | " + " | ".join(LABELS[m] for m in METRICS) + " |"
    sep = "|---" * (len(METRICS) + 1) + "|"
    rows = [header, sep]
    for slice_name, metrics in report.items():
        cells = [
            f"{fmt(metrics[m].mean)} ({metrics[m].n})" if metrics[m].n else "–" for m in METRICS
        ]
        rows.append(f"| {slice_name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def mode_comparison_table(reports: Mapping[str, Report]) -> str:
    """Rows = retrieval modes, columns = metrics, global slice only."""
    header = "| Retriever | " + " | ".join(LABELS[m] for m in METRICS) + " |"
    sep = "|---" * (len(METRICS) + 1) + "|"
    rows = [header, sep]
    for mode, report in reports.items():
        overall = report.get("all", {})
        cells = [fmt(overall[m].mean) if m in overall else "–" for m in METRICS]
        rows.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
