"""Log eval runs to MLflow (local SQLite store, no server needed)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from uae_rag.config import Settings
from uae_rag.evaluation.metrics import Report


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
        ).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def flatten(prefix: str, value: object, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = str(value)[:250]


def metric_key(slice_name: str, metric: str) -> str:
    return f"{metric}.{slice_name}"


def log_run(
    settings: Settings,
    mode: str,
    report: Report,
    artifacts_dir: Path,
    tags: dict[str, str],
    root: Path,
) -> str | None:
    import mlflow

    mlflow.set_tracking_uri(settings.paths.mlflow_uri)
    mlflow.set_experiment("stage1-eval")
    params: dict[str, str] = {}
    flatten("", settings.model_dump(mode="json", exclude={"paths"}), params)
    params["retrieval.mode"] = mode
    with mlflow.start_run(run_name=f"{mode}-{tags.get('dataset', '')}") as run:
        mlflow.set_tags({"git_sha": git_sha(root), **tags})
        mlflow.log_params(params)
        for slice_name, metrics in report.items():
            for name, summary in metrics.items():
                if summary.mean is not None:
                    mlflow.log_metric(metric_key(slice_name, name), summary.mean)
        mlflow.log_artifacts(str(artifacts_dir))
        return run.info.run_id
