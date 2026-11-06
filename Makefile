# Works with GNU make on Linux/macOS and on Windows (e.g. `winget install ezwinports.make`).
# The virtualenv lives outside the repo by default so cloud-sync folders don't upload it.

ifeq ($(OS),Windows_NT)
  VENV ?= $(subst \,/,$(USERPROFILE))/.venvs/uae-regulatory-rag
  PY := $(VENV)/Scripts/python.exe
  BOOTSTRAP ?= py -3.11
else
  VENV ?= .venv
  PY := $(VENV)/bin/python
  BOOTSTRAP ?= python3.11
endif
CLI := $(PY) -m uae_rag.cli
MODES ?= dense,hybrid,hybrid_rerank

.PHONY: help setup setup-pip lock download ingest index ask draft-golden verify-golden ci-subset eval eval-fast \
        baseline gate calibrate-label calibrate-report ui demo mlflow test lint format

help:
	@echo "setup | download | ingest | ask Q='...' | ui | draft-golden | verify-golden | ci-subset"
	@echo "eval | eval-fast | baseline | gate RUN=eval/runs/<dir> | calibrate-label ITEMS=... | calibrate-report ITEMS=..."
	@echo "mlflow | test | lint | format"

# `uv` is much faster than pip and reuses packages already in its cache
# (e.g. torch from another project). Install it once with: $(BOOTSTRAP) -m pip install --user uv
UV := $(BOOTSTRAP) -m uv
TORCH_CPU := https://download.pytorch.org/whl/cpu

setup:
	$(BOOTSTRAP) -m venv "$(VENV)"
	$(UV) pip install --python "$(PY)" -r requirements.txt --extra-index-url $(TORCH_CPU) --index-strategy unsafe-best-match
	$(UV) pip install --python "$(PY)" --no-deps -e .

setup-pip:  ## fallback without uv (slower)
	$(BOOTSTRAP) -m venv "$(VENV)"
	$(PY) -m pip install -r requirements.txt --extra-index-url $(TORCH_CPU)
	$(PY) -m pip install -e . --no-deps

lock:  ## re-pin requirements.txt (packages at least 2 weeks old; torch pinned explicitly)
	$(UV) pip compile pyproject.toml --extra dev --universal --python-version 3.11 \
	  --exclude-newer $(LOCK_DATE) --exclude-newer-package torch=$(TORCH_DATE) -o requirements.txt
LOCK_DATE ?= 2026-10-23
TORCH_DATE ?= 2026-11-06

download:
	$(CLI) download

ingest:
	$(CLI) ingest

index:  ## re-embed data/chunks.jsonl only (no PDF parsing)
	$(CLI) index

ask:
	$(CLI) ask "$(Q)"

ui:
	$(PY) -m streamlit run src/uae_rag/ui/app.py

draft-golden:
	$(CLI) draft-golden --n 120

verify-golden:
	$(CLI) verify-golden

ci-subset:
	$(CLI) ci-subset --n 15

eval:
	$(CLI) eval --modes $(MODES)

eval-fast:  ## retrieval metrics only: no LLM calls
	$(CLI) eval --modes $(MODES) --no-generate --tag retrieval

baseline:
	$(CLI) eval --golden eval/golden_ci.jsonl --modes hybrid_rerank --save-baseline --tag baseline

gate:
	$(CLI) gate $(RUN) --comment-out gate_comment.md

calibrate-label:
	$(CLI) calibrate-label $(ITEMS)

calibrate-report:
	$(CLI) calibrate-report $(ITEMS)

demo:  ## record docs/demo.webm + docs/demo.gif (needs `make ui` running; pip install -e ".[demo]")
	$(PY) scripts/record_demo.py --url http://localhost:8501 --out docs

mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests
