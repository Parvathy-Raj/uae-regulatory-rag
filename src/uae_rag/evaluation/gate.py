"""CI release gate: compare a run against eval/baseline.json."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from uae_rag.evaluation.metrics import Report
from uae_rag.evaluation.report import LABELS, fmt


class GateConfig(BaseModel):
    mode: str = "hybrid_rerank"
    max_drop: dict[str, float] = Field(
        default_factory=lambda: {"recall_at_5": 0.05, "faithfulness": 0.05}
    )
    min_items: int = 3


class GateCheck(BaseModel):
    slice: str
    metric: str
    baseline: float | None
    current: float | None
    n: int
    passed: bool
    note: str = ""


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck]
    message: str = ""


def load_gate_config(path: Path) -> GateConfig:
    return GateConfig(**(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))


def report_to_plain(report: Report) -> dict[str, dict[str, dict]]:
    return {s: {m: v.model_dump() for m, v in ms.items()} for s, ms in report.items()}


def evaluate_gate(current: dict, baseline: dict | None, cfg: GateConfig) -> GateResult:
    """current/baseline are plain dicts: slice -> metric -> {"mean": float|None, "n": int}."""
    if baseline is None:
        return GateResult(
            passed=True,
            checks=[],
            message="No baseline yet: gate skipped. Run `make baseline` on main.",
        )
    checks: list[GateCheck] = []
    for slice_name, metrics in current.items():
        for metric, max_drop in cfg.max_drop.items():
            cur = metrics.get(metric, {})
            base = baseline.get(slice_name, {}).get(metric, {})
            cur_mean, base_mean, n = cur.get("mean"), base.get("mean"), int(cur.get("n", 0))
            if cur_mean is None or base_mean is None:
                continue
            if slice_name != "all" and n < cfg.min_items:
                checks.append(
                    GateCheck(
                        slice=slice_name,
                        metric=metric,
                        baseline=base_mean,
                        current=cur_mean,
                        n=n,
                        passed=True,
                        note="too few items",
                    )
                )
                continue
            ok = cur_mean >= base_mean - max_drop
            checks.append(
                GateCheck(
                    slice=slice_name,
                    metric=metric,
                    baseline=base_mean,
                    current=cur_mean,
                    n=n,
                    passed=ok,
                    note="" if ok else f"dropped more than {max_drop:.2f}",
                )
            )
    return GateResult(passed=all(c.passed for c in checks), checks=checks)


def gate_markdown(result: GateResult, category_table_md: str, header: str = "") -> str:
    status = "✅ **Eval gate passed**" if result.passed else "❌ **Eval gate failed**"
    lines = ["<!-- uae-rag-eval-gate -->", header, status, ""]
    if result.message:
        lines += [result.message, ""]
    if result.checks:
        lines += [
            "| Slice | Metric | Baseline | This PR | Δ | n | Result |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in result.checks:
            delta = (c.current or 0) - (c.baseline or 0)
            mark = "✅" if c.passed else "❌"
            lines.append(
                f"| {c.slice} | {LABELS.get(c.metric, c.metric)} | {fmt(c.baseline)} | "
                f"{fmt(c.current)} | {delta:+.3f} | {c.n} | {mark} {c.note} |"
            )
        lines.append("")
    lines += [
        "<details><summary>Per-category results (this PR)</summary>",
        "",
        category_table_md,
        "",
        "</details>",
    ]
    return "\n".join(line for line in lines if line is not None)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
