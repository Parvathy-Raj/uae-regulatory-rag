"""Typed settings loaded from config/settings.yaml, corpus.yaml and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl

ROOT = Path(__file__).resolve().parents[2]
RetrievalMode = Literal["dense", "hybrid", "hybrid_rerank"]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "hybrid", "hybrid_rerank")


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str = ""


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    judge_model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    max_tokens: int = 1024
    reasoning_effort: str = ""  # "low" | "medium" | "high" for reasoning models; "" = not sent
    timeout_s: float = 120
    max_retries: int = 3
    requests_per_minute: int = 25
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    def provider_config(self) -> ProviderConfig:
        if self.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "none"
            raise ValueError(f"Unknown LLM provider '{self.provider}'. Configured: {known}")
        return self.providers[self.provider]

    def api_key(self) -> str:
        env = self.provider_config().api_key_env
        return os.getenv(env, "").strip() if env else ""


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = ""
    batch_size: int = 32


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-base"


class ChunkingConfig(BaseModel):
    max_words: int = Field(300, gt=10)
    overlap_words: int = Field(50, ge=0)
    min_words: int = Field(40, ge=0)


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = "hybrid_rerank"
    top_k: int = Field(5, gt=0)
    candidate_k: int = Field(30, gt=0)
    rrf_k: int = Field(60, gt=0)


class GenerationConfig(BaseModel):
    min_rerank_score: float = 0.0
    min_dense_score: float = 0.0
    refuse_if_no_valid_citation: bool = True


class PathsConfig(BaseModel):
    corpus_file: Path = Path("corpus.yaml")
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    chunks_file: Path = Path("data/chunks.jsonl")
    chroma_dir: Path = Path("data/chroma")
    collection: str = "fta_corpus"
    golden_file: Path = Path("eval/golden.jsonl")
    golden_ci_file: Path = Path("eval/golden_ci.jsonl")
    baseline_file: Path = Path("eval/baseline.json")
    runs_dir: Path = Path("eval/runs")
    mlflow_uri: str = "sqlite:///mlruns.db"

    def resolve(self, root: Path) -> PathsConfig:
        """Make every relative path absolute against the repo root."""
        data = self.model_dump()
        for key, value in data.items():
            if isinstance(value, Path) and not value.is_absolute():
                data[key] = root / value
        return PathsConfig(**data)


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


class CorpusSource(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    domain: str
    url: HttpUrl


class Corpus(BaseModel):
    name: str
    publisher: str = ""
    sources: list[CorpusSource]


def _apply_env(settings: Settings) -> Settings:
    overrides = {
        "provider": os.getenv("UAE_RAG_LLM_PROVIDER"),
        "model": os.getenv("UAE_RAG_LLM_MODEL"),
        "judge_model": os.getenv("UAE_RAG_JUDGE_MODEL"),
    }
    clean = {k: v.strip() for k, v in overrides.items() if v and v.strip()}
    if clean:
        settings.llm = settings.llm.model_copy(update=clean)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()
    if ollama_url and "ollama" in settings.llm.providers:
        settings.llm.providers["ollama"] = ProviderConfig(base_url=ollama_url)
    return settings


def load_settings(path: Path | None = None, root: Path = ROOT) -> Settings:
    """Load settings YAML (default config/settings.yaml), apply env overrides, resolve paths."""
    _load_dotenv(root)
    path = path or Path(os.getenv("UAE_RAG_SETTINGS", root / "config" / "settings.yaml"))
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    settings = _apply_env(Settings(**raw))
    settings.paths = settings.paths.resolve(root)
    return settings


def load_corpus(path: Path) -> Corpus:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    corpus = Corpus(**raw)
    ids = [s.id for s in corpus.sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source ids in corpus file")
    return corpus


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(root / ".env", override=False)
